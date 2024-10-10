import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, FastMarkerCluster
from streamlit_folium import st_folium
import sqlite3
from sqlalchemy import create_engine, Table, Column, String, Float, MetaData, select, inspect, text, DateTime
from sqlalchemy.exc import SQLAlchemyError
from geopy.geocoders import GoogleV3, Here, Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderQuotaExceeded
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import time
import os
from datetime import datetime, timedelta
import threading

# ------------------------ Logging Configuration ------------------------
LOG_FILE = "geocoding.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

# ------------------------ Quota Management ------------------------
class GeocoderQuotaManager:
    def __init__(self, max_quota):
        self.max_quota = max_quota
        self.lock = threading.Lock()
        self.calls_made = 0

    def can_make_call(self):
        with self.lock:
            return self.calls_made < self.max_quota

    def record_call(self):
        with self.lock:
            self.calls_made += 1

    def remaining_quota(self):
        with self.lock:
            return self.max_quota - self.calls_made

# ------------------------ Database Functions ------------------------

@st.cache_resource
def init_db(db_name='geocoded_addresses.db'):
    """
    Initializes the SQLite database and creates the 'geocoded' table if it doesn't exist.
    Utilizes connection pooling for better performance.
    """
    logging.debug("Initializing database with connection pooling.")
    try:
        engine = create_engine(
            f'sqlite:///{db_name}',
            connect_args={"check_same_thread": False},
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
        metadata = MetaData()
        geocoded_table = Table(
            'geocoded', metadata,
            Column('address', String, primary_key=True, index=True),
            Column('latitude', Float, index=True),
            Column('longitude', Float, index=True),
            Column('timestamp', DateTime, default=datetime.utcnow, index=True)
        )
        metadata.create_all(engine)
        logging.info("Database and table 'geocoded' initialized.")

        # Ensure 'timestamp' column uses DateTime
        inspector = inspect(engine)
        columns = [column['name'] for column in inspector.get_columns('geocoded')]
        if 'timestamp' not in columns:
            with engine.connect() as conn:
                try:
                    conn.execute(text('ALTER TABLE geocoded ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP'))
                    logging.info("Added 'timestamp' column to 'geocoded' table.")
                except SQLAlchemyError as e:
                    st.error(f"Error adding 'timestamp' column: {e}")
                    logging.error(f"Error adding 'timestamp' column: {e}")
        else:
            logging.debug("'timestamp' column already exists in 'geocoded' table.")

        return engine, geocoded_table
    except Exception as e:
        logging.critical(f"Failed to initialize database: {e}")
        st.error(f"Failed to initialize database: {e}")
        raise

@st.cache_data
def fetch_cached_addresses(_engine, _table, addresses, ttl_days=30):
    """
    Fetches cached geocoded addresses from the database that are not older than ttl_days.
    """
    logging.debug("Fetching cached addresses from the database with TTL filtering.")
    try:
        with _engine.connect() as conn:
            cutoff_date = datetime.utcnow() - timedelta(days=ttl_days)
            query = _table.select().where(
                (_table.c.address.in_(addresses)) &
                (_table.c.timestamp >= cutoff_date)
            )
            result = conn.execute(query).fetchall()
            df = pd.DataFrame(result, columns=result[0].keys()) if result else pd.DataFrame(columns=['address', 'latitude', 'longitude', 'timestamp'])
            logging.info(f"Fetched {len(df)} cached addresses within TTL.")
            return df
    except Exception as e:
        logging.error(f"Error fetching cached addresses: {e}")
        st.error(f"Error fetching cached addresses: {e}")
        return pd.DataFrame(columns=['address', 'latitude', 'longitude', 'timestamp'])

def insert_geocoded_addresses(_engine, _table, data):
    """
    Inserts geocoded addresses into the database in bulk.
    """
    logging.debug("Inserting geocoded addresses into the database.")
    try:
        with _engine.begin() as conn:
            conn.execute(_table.insert(), [
                {
                    'address': entry['address'],
                    'latitude': entry['latitude'],
                    'longitude': entry['longitude'],
                    'timestamp': datetime.utcnow()
                }
                for entry in data
            ])
        logging.info(f"Inserted {len(data)} geocoded addresses into the database.")
    except Exception as e:
        logging.error(f"Error inserting geocoded addresses: {e}")
        st.error(f"Error inserting geocoded addresses: {e}")

# ------------------------ Geocoding Functions ------------------------

def get_geolocator(preferred_providers, api_keys=None):
    """
    Initializes and returns a geolocator based on the preferred providers' order.
    """
    logging.debug(f"Initializing geolocator with preferred providers: {preferred_providers}")
    for provider in preferred_providers:
        try:
            if provider == "Nominatim":
                geolocator = Nominatim(user_agent="myGeocoder", timeout=10)
                logging.info("Nominatim geolocator initialized.")
                return geolocator
            elif provider == "Here":
                api_key = api_keys.get("Here")
                if api_key:
                    geolocator = Here(apikey=api_key, timeout=10)
                    logging.info("Here geolocator initialized.")
                    return geolocator
            elif provider == "Google":
                api_key = api_keys.get("Google")
                if api_key:
                    geolocator = GoogleV3(api_key=api_key, timeout=10)
                    logging.info("GoogleV3 geolocator initialized.")
                    return geolocator
        except Exception as e:
            logging.error(f"Error initializing {provider} geolocator: {e}")
            continue  # Try the next provider
    logging.error("No valid geolocators could be initialized.")
    return None

def estimate_geocoding_cost(provider, num_addresses):
    """
    Estimates the cost of geocoding based on the provider and number of addresses.
    """
    pricing = {
        "Google": 0.005,  # Example cost per geocode in USD
        "Here": 0.004,
        "Nominatim": 0.0  # Free
    }
    cost_per_geocode = pricing.get(provider, 0)
    return cost_per_geocode * num_addresses

def geocode_address_sync(geolocator, address, retries=3, backoff=2, cancel_event=None, quota_manager=None, lock=threading.Lock()):
    """
    Synchronously geocodes a single address with retry logic and rate limiting.
    """
    logging.debug(f"Starting geocoding for address: {address}")
    for attempt in range(retries):
        if cancel_event and cancel_event.is_set():
            logging.info(f"Geocoding cancelled by user for address: {address}")
            return {'address': address, 'latitude': None, 'longitude': None}
        if quota_manager and not quota_manager.can_make_call():
            logging.error("Geocoding API quota reached.")
            st.error("Geocoding API quota reached.")
            return {'address': address, 'latitude': None, 'longitude': None}
        try:
            with lock:  # Ensure rate limiting by controlling access
                location = geolocator.geocode(address)
            if quota_manager:
                quota_manager.record_call()
            if location:
                logging.info(f"Geocoded address: {address} to ({location.latitude}, {location.longitude})")
                return {
                    'address': address,
                    'latitude': location.latitude,
                    'longitude': location.longitude
                }
            else:
                logging.warning(f"Geocoding failed: No location found for address: {address}")
                return {'address': address, 'latitude': None, 'longitude': None}
        except GeocoderQuotaExceeded:
            logging.error("Geocoding API quota exceeded.")
            st.error("Geocoding API quota exceeded.")
            return {'address': address, 'latitude': None, 'longitude': None}
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            wait_time = backoff ** attempt + 0.1 * attempt
            logging.warning(f"Geocoding failed for address: {address}. Attempt {attempt +1} of {retries}. Waiting {wait_time:.2f} seconds. Error: {e}")
            time.sleep(wait_time)
        except Exception as e:
            logging.error(f"Unexpected error during geocoding for address: {address} - {e}")
            return {'address': address, 'latitude': None, 'longitude': None}
    logging.warning(f"All geocoding attempts failed for address: {address}")
    return {'address': address, 'latitude': None, 'longitude': None}

def geocode_addresses_sync(geolocator, addresses, engine, table, max_workers=10, cancel_event=None, quota_manager=None):
    """
    Geocodes a list of addresses synchronously using ThreadPoolExecutor with quota management.
    """
    logging.debug(f"Starting synchronous geocoding for {len(addresses)} addresses with max_workers={max_workers}.")
    all_results = []
    all_failed = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_address = {
            executor.submit(geocode_address_sync, geolocator, addr, cancel_event=cancel_event, quota_manager=quota_manager, lock=lock): addr for addr in addresses
        }

        progress_bar = st.progress(0)
        total = len(future_to_address)
        completed = 0

        for future in as_completed(future_to_address):
            addr = future_to_address[future]
            try:
                result = future.result()
                if result['latitude'] is not None and result['longitude'] is not None:
                    all_results.append(result)
                else:
                    all_failed.append(result['address'])
            except Exception as e:
                logging.error(f"Error geocoding address {addr}: {e}")
                all_failed.append(addr)
            completed += 1
            progress_bar.progress(completed / total)
            logging.debug(f"Geocoding progress: {completed}/{total} addresses processed.")

    if all_results:
        insert_geocoded_addresses(engine, table, all_results)
        logging.info(f"Inserted {len(all_results)} new geocoded addresses into the database.")

    logging.debug(f"Geocoding completed with {len(all_results)} successful and {len(all_failed)} failed addresses.")
    return pd.DataFrame(all_results), all_failed

# ------------------------ Heatmap Functions ------------------------

def create_h3_heatmap(df, resolution=7):
    """
    Creates an H3 heatmap DataFrame from geocoded addresses.
    """
    logging.debug(f"Creating H3 heatmap with resolution {resolution}.")
    try:
        import h3
        import geopandas as gpd
    except ImportError as e:
        logging.error(f"Missing dependencies for H3 heatmap: {e}")
        st.error(f"Missing dependencies for H3 heatmap: {e}")
        return pd.DataFrame()

    try:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
        gdf['h3_index'] = gdf.geometry.apply(lambda p: h3.geo_to_h3(p.y, p.x, resolution))
        h3_counts = gdf['h3_index'].value_counts().reset_index()
        h3_counts.columns = ['h3_index', 'count']
        h3_counts[['lat', 'lng']] = h3_counts['h3_index'].apply(lambda h: pd.Series(h3.h3_to_geo(h)))
        logging.info(f"H3 heatmap created with {len(h3_counts)} hexagons.")
        return h3_counts
    except Exception as e:
        logging.error(f"Error creating H3 heatmap: {e}")
        st.error(f"Error creating H3 heatmap: {e}")
        return pd.DataFrame()

# ------------------------ Main Function ------------------------

def main():
    logging.debug("Starting main function.")
    st.set_page_config(page_title="📍 Optimized Address Heatmap Generator", layout="wide")
    st.title("📍 Optimized Address Heatmap Generator")

    st.markdown("""
    **Generate interactive heatmaps from large address lists efficiently!**
    
    Upload your addresses (up to 10,000), geocode them using various providers, and visualize the distribution on an interactive map.
    """)

    # Initialize session state for cancellation
    if 'cancel_geocoding' not in st.session_state:
        st.session_state['cancel_geocoding'] = False
        logging.debug("Initialized session state for 'cancel_geocoding'.")

    # Initialize Database
    try:
        engine, table = init_db()
    except Exception as e:
        logging.critical(f"Exiting application due to database initialization failure: {e}")
        return

    # Sidebar for customization options
    st.sidebar.header("🔧 Heatmap Customization")
    radius = st.sidebar.slider("Radius", min_value=1, max_value=50, value=25, step=1)
    blur = st.sidebar.slider("Blur", min_value=1, max_value=50, value=15, step=1)
    min_opacity = st.sidebar.slider("Minimum Opacity", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    st.sidebar.subheader("🎨 Gradient Colors")
    color1 = st.sidebar.color_picker("Color 1", "#0f0")
    color2 = st.sidebar.color_picker("Color 2", "#f00")
    color3 = st.sidebar.color_picker("Color 3", "#00f")
    gradient = {0.0: color1, 0.5: color2, 1.0: color3}
    logging.debug(f"Gradient colors set to: {gradient}")

    st.sidebar.subheader("🗺️ Map Settings")
    map_style = st.sidebar.selectbox(
        "Select Map Style",
        options=["OpenStreetMap", "Stamen Terrain", "Stamen Toner", "CartoDB Positron", "CartoDB Dark_Matter"]
    )
    logging.debug(f"Map style selected: {map_style}")

    st.sidebar.markdown("---")

    # Address input options
    st.subheader("1️⃣ Input Addresses")
    input_method = st.radio("Choose input method:", ("Upload Address File", "Manual Entry"))
    logging.info(f"Input method selected: {input_method}")

    addresses = []
    if input_method == "Upload Address File":
        uploaded_file = st.file_uploader("Upload a file containing addresses", type=["txt", "csv", "xlsx", "parquet"])
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            logging.debug(f"Uploaded file detected with extension: {file_extension}")
            try:
                if file_extension == "txt":
                    addresses = [line.decode('utf-8').strip() for line in uploaded_file if line.decode('utf-8').strip()]
                elif file_extension == "csv":
                    df = pd.read_csv(uploaded_file)
                    addresses = df.iloc[:,0].dropna().astype(str).str.strip().tolist()
                elif file_extension == "xlsx":
                    df = pd.read_excel(uploaded_file)
                    addresses = df.iloc[:,0].dropna().astype(str).str.strip().tolist()
                elif file_extension == "parquet":
                    df = pq.read_table(uploaded_file).to_pandas()
                    addresses = df.iloc[:,0].dropna().astype(str).str.strip().tolist()
                # Remove duplicates and empty entries
                addresses = list(set([addr for addr in addresses if addr]))
                st.success(f"Loaded {len(addresses)} unique addresses.")
                logging.info(f"Loaded {len(addresses)} unique addresses from uploaded file.")
            except Exception as e:
                st.error(f"Error processing the file: {e}")
                logging.error(f"Error processing the uploaded file: {e}")
    else:
        manual_addresses = st.text_area("Enter addresses manually, one per line:")
        if manual_addresses:
            addresses = list(set([line.strip() for line in manual_addresses.splitlines() if line.strip()]))
            st.success(f"Loaded {len(addresses)} unique addresses.")
            logging.info(f"Loaded {len(addresses)} unique addresses from manual entry.")

    # Set maximum address limit
    MAX_ADDRESSES = 10000  # Define your maximum limit
    if len(addresses) > MAX_ADDRESSES:
        st.error(f"Number of addresses exceeds the maximum limit of {MAX_ADDRESSES}. Please reduce your input.")
        addresses = addresses[:MAX_ADDRESSES]
        logging.warning(f"Address input limited to the first {MAX_ADDRESSES} addresses.")

    if addresses:
        st.subheader("2️⃣ Generate Heatmap")
        # Define provider priority
        preferred_order = ["Google", "Here", "Nominatim"]  # Define your preferred order

        # Collect API keys from secrets or user input
        api_keys = {
            "Google": st.secrets.get("google", {}).get("api_key") if "google" in st.secrets else None,
            "Here": st.secrets.get("here", {}).get("api_key") if "here" in st.secrets else None,
        }

        # Allow manual input of API keys if not in secrets
        if "Google" in preferred_order and not api_keys["Google"]:
            api_keys["Google"] = st.text_input("Enter your Google Geocoding API key:", type="password")
            logging.debug("Google API key entered manually.")
        if "Here" in preferred_order and not api_keys["Here"]:
            api_keys["Here"] = st.text_input("Enter your Here Geocoding API key:", type="password")
            logging.debug("Here API key entered manually.")

        geolocator = get_geolocator(preferred_order, api_keys=api_keys)

        if not geolocator:
            st.error("Failed to initialize any geocoding provider. Please check your API keys.")
            logging.error("Failed to initialize any geocoding provider.")
            return

        # Define max_workers based on provider
        geocoding_provider = preferred_order[0]  # Current provider in use
        if geocoding_provider == "Nominatim":
            max_workers = 1  # Nominatim allows 1 request per second per IP by default
        elif geocoding_provider == "Here":
            max_workers = st.sidebar.slider(
                "Max Concurrent Requests", min_value=1, max_value=20, value=10, step=1
            )
        elif geocoding_provider == "Google":
            max_workers = st.sidebar.slider(
                "Max Concurrent Requests", min_value=1, max_value=20, value=10, step=1
            )
        else:
            max_workers = 5  # Default value
        logging.debug(f"Max concurrent requests set to: {max_workers}")

        # Estimate geocoding cost
        new_addresses = addresses  # Initially, all are new; will adjust after fetching cache
        estimated_cost = estimate_geocoding_cost(geocoding_provider, len(new_addresses))
        st.info(f"Estimated geocoding cost for {len(new_addresses)} addresses using {geocoding_provider}: ${estimated_cost:.2f}")
        YOUR_COST_THRESHOLD = 100.0  # Define your cost threshold
        if geocoding_provider in ["Google", "Here"] and estimated_cost > YOUR_COST_THRESHOLD:
            st.warning("The estimated cost exceeds your predefined threshold. Please review your input.")
            logging.warning("Estimated cost exceeds the predefined threshold.")

        if st.button("🔍 Generate Heatmap"):
            logging.info("Generate Heatmap button clicked.")
            if geocoding_provider in ["Google", "Here"] and not api_keys.get(geocoding_provider):
                st.error(f"{geocoding_provider} API key is required.")
                logging.error(f"{geocoding_provider} API key not provided.")
            elif geocoding_provider not in ["Google", "Here"] and not geolocator:
                st.error("Geolocator is not configured correctly.")
                logging.error("Geolocator initialization failed.")
            else:
                with st.spinner("Starting geocoding process..."):
                    st.session_state['cancel_geocoding'] = False
                    logging.info("Geocoding process started.")
                    try:
                        # Fetch cached addresses
                        cached = fetch_cached_addresses(_engine=engine, _table=table, addresses=addresses)
                        cached_addresses = set(cached['address'])
                        # Implement TTL filtering
                        ttl_days = 30  # You can make this configurable if needed
                        cutoff_date = datetime.utcnow() - timedelta(days=ttl_days)
                        valid_cached = cached[cached['timestamp'] >= cutoff_date]
                        cached_addresses = set(valid_cached['address'])
                        new_addresses = [addr for addr in addresses if addr not in cached_addresses]
                        logging.debug(f"{len(new_addresses)} addresses to geocode after caching and TTL filtering.")

                        # Update estimated cost based on new_addresses
                        estimated_cost = estimate_geocoding_cost(geocoding_provider, len(new_addresses))
                        st.info(f"Actual geocoding cost for {len(new_addresses)} new addresses using {geocoding_provider}: ${estimated_cost:.2f}")
                        if geocoding_provider in ["Google", "Here"] and estimated_cost > YOUR_COST_THRESHOLD:
                            st.warning("The actual estimated cost exceeds your predefined threshold. Please review your input.")
                            logging.warning("Actual estimated cost exceeds the predefined threshold.")

                        st.write(f"🔍 Geocoding {len(new_addresses)} new addresses...")
                        logging.info(f"Geocoding {len(new_addresses)} new addresses.")

                        if new_addresses:
                            # Define quota based on provider
                            provider_quota = {
                                "Google": 50000,  # Example quota
                                "Here": 100000,
                                "Nominatim": 1000  # Nominatim has strict usage policies
                            }
                            quota_manager = GeocoderQuotaManager(max_quota=provider_quota.get(geocoding_provider, 0))

                            # Run synchronous geocoding
                            cache, failed = geocode_addresses_sync(
                                geolocator, new_addresses, engine, table, max_workers=max_workers, cancel_event=None, quota_manager=quota_manager
                            )
                            logging.debug(f"Geocoding sync completed. Success: {len(cache)}, Failed: {len(failed)}")
                        else:
                            cache = cached
                            failed = []
                            logging.info("No new addresses to geocode; using cached data.")

                    except Exception as e:
                        st.error(f"An error occurred during geocoding: {e}")
                        logging.error(f"Geocoding process failed: {e}")
                        return

            if cache.empty and not failed:
                st.error("No addresses were geocoded successfully.")
                logging.warning("No addresses were geocoded successfully.")
                return

            st.success("✅ Geocoding completed. Generating heatmap...")
            logging.info("Geocoding completed. Proceeding to generate heatmap.")

            # Combine cached and new results
            if not cache.empty and not cached.empty:
                cache = pd.concat([valid_cached, cache], ignore_index=True)
                logging.debug("Combined cached and new geocoded addresses.")
            elif not cached.empty:
                cache = valid_cached
                logging.debug("Using only cached geocoded addresses.")
            # Else, use cache as is

            # Ensure there is data to plot
            if cache.empty:
                st.error("No data available to generate the heatmap.")
                logging.error("No data available to generate the heatmap.")
                return

            # Create H3 heatmap
            h3_resolution = st.slider("H3 Resolution", min_value=1, max_value=15, value=7, step=1)
            logging.debug(f"H3 resolution set to: {h3_resolution}")
            h3_heatmap = create_h3_heatmap(cache, resolution=h3_resolution)

            if h3_heatmap.empty:
                st.error("Failed to create H3 heatmap.")
                logging.error("Failed to create H3 heatmap.")
                return

            # Create Folium Map
            try:
                mean_lat = cache['latitude'].mean()
                mean_lng = cache['longitude'].mean()
                logging.debug(f"Map center set to: ({mean_lat}, {mean_lng})")
                m = folium.Map(location=[mean_lat, mean_lng], zoom_start=5, tiles=map_style, control_scale=True)
                logging.info("Folium map initialized.")

                # Add HeatMap Layer
                heat_data = h3_heatmap[['lat', 'lng', 'count']].values.tolist()
                heatmap_layer = HeatMap(
                    heat_data,
                    name='Heatmap',
                    radius=radius,
                    blur=blur,
                    min_opacity=min_opacity,
                    gradient=gradient
                )
                heatmap_layer.add_to(m)
                logging.info("HeatMap layer added to Folium map.")

                # Add Marker Layer using FastMarkerCluster
                marker_data = cache[['latitude', 'longitude']].values.tolist()
                FastMarkerCluster(marker_data, name='Markers').add_to(m)
                logging.info("FastMarkerCluster layer added to Folium map.")

                # Add Layer Control
                folium.LayerControl().add_to(m)
                logging.debug("LayerControl added to Folium map.")

            except Exception as e:
                st.error(f"Error creating the map: {e}")
                logging.error(f"Error creating the Folium map: {e}")
                return

            # Display Map
            try:
                st.subheader("3️⃣ Generated Heatmap")
                map_container = st.container()
                with map_container:
                    logging.debug("Displaying Folium map in Streamlit.")
                    st_folium(m, width=700, height=500, returned_objects=None)
                logging.info("Folium map displayed successfully.")
            except Exception as e:
                st.error(f"Error displaying the map: {e}")
                logging.error(f"Error displaying the Folium map: {e}")
                return

            # Download Heatmap and Data
            try:
                with st.expander("📥 Download Heatmap and Data"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
                            m.save(tmpfile.name)
                            logging.debug("Heatmap saved to temporary HTML file.")
                            with open(tmpfile.name, "rb") as f:
                                st.download_button(
                                    label="💾 Download Heatmap as HTML",
                                    data=f,
                                    file_name="heatmap.html",
                                    mime="text/html"
                                )
                        logging.info("Heatmap HTML download button created.")
                    with col2:
                        # Download Geocoded Data as Parquet
                        st.markdown("### 📥 Download Geocoded Data")
                        geocoded_data = cache[['address', 'latitude', 'longitude']]

                        parquet_buffer = pa.BufferOutputStream()
                        pq.write_table(pa.Table.from_pandas(geocoded_data), parquet_buffer)
                        st.download_button(
                            label="Download as Parquet",
                            data=parquet_buffer.getvalue().to_pybytes(),
                            file_name='geocoded_addresses.parquet',
                            mime='application/octet-stream',
                        )
                        logging.info("Geocoded data Parquet download button created.")
                    with col3:
                        # Download Geocoded Data as CSV
                        csv = geocoded_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name='geocoded_addresses.csv',
                            mime='text/csv',
                        )
                        logging.info("Geocoded data CSV download button created.")

                    # Handle Failed Addresses
                    if failed:
                        st.warning("⚠️ Some addresses could not be geocoded:")
                        st.write(failed[:50])  # Display first 50 failed addresses
                        if len(failed) > 50:
                            st.write(f"...and {len(failed) - 50} more.")
                        logging.warning(f"{len(failed)} addresses failed to geocode.")

                        # Option to Download Failed Addresses
                        failed_df = pd.DataFrame({'failed_addresses': failed})
                        csv_failed = failed_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Download Failed Addresses as CSV",
                            data=csv_failed,
                            file_name='failed_addresses.csv',
                            mime='text/csv',
                        )
                        logging.info("Failed addresses CSV download button created.")
            except Exception as e:
                st.error(f"Error creating download options: {e}")
                logging.error(f"Error creating download buttons: {e}")

        # Option to Cancel Geocoding
        # Note: Implementing cancellation for synchronous code is cooperative.
        st.sidebar.markdown("---")
        if st.sidebar.button("❌ Cancel Geocoding"):
            st.session_state['cancel_geocoding'] = True
            st.sidebar.success("Geocoding process has been cancelled.")
            logging.info("User initiated cancellation of geocoding process.")
        st.sidebar.markdown("© 2024 Your Company Name")

    logging.debug("Main function completed.")

if __name__ == "__main__":
    main()
    logging.info("Application finished.")
