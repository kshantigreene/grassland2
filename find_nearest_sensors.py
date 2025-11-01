import csv
from geopy.distance import geodesic

def load_sensor_locations(file_path):
    """Load sensor locations from CSV file."""
    sensors = {}
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sensor_id = row['Sensor']
            lat = row['lat'].strip()
            lon = row['lon'].strip()
            if lat and lon:  # Only add if coordinates exist
                sensors[sensor_id] = (float(lat), float(lon))
    return sensors

def load_nest_locations(file_path):
    """Load nest locations from CSV file."""
    nests = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nest_id = row['Nest_id']
            lat = row['lat'].strip() if row['lat'] else ''
            lon = row['lon'].strip() if row['lon'] else ''
            species = row['Species_code']

            nests.append({
                'Nest_id': nest_id,
                'lat': lat,
                'lon': lon,
                'Species_code': species,
                '1st_Closest': row.get('1st_Closest', ''),
                '2nd_closest': row.get('2nd_closest', ''),
                '3rd_closest': row.get('3rd_closest', '')
            })
    return nests

def find_nearest_sensors(nest_coords, sensors):
    """Find the 3 nearest sensors to a nest location."""
    if not nest_coords[0] or not nest_coords[1]:
        return [], []

    distances = []
    for sensor_id, sensor_coords in sensors.items():
        distance = geodesic(nest_coords, sensor_coords).km
        distances.append((sensor_id, distance))

    # Sort by distance and get top 3
    distances.sort(key=lambda x: x[1])
    top_3 = distances[:3]

    sensor_ids = [sensor_id for sensor_id, _ in top_3]
    sensor_distances = [distance for _, distance in top_3]

    return sensor_ids, sensor_distances

def update_nest_locations(nests, sensors):
    """Update nest locations with nearest sensors."""
    for nest in nests:
        if nest['lat'] and nest['lon']:
            nest_coords = (float(nest['lat']), float(nest['lon']))
            sensor_ids, distances = find_nearest_sensors(nest_coords, sensors)

            # Update the nest dictionary with nearest sensors
            if len(sensor_ids) >= 1:
                nest['1st_Closest'] = sensor_ids[0]
            if len(sensor_ids) >= 2:
                nest['2nd_closest'] = sensor_ids[1]
            if len(sensor_ids) >= 3:
                nest['3rd_closest'] = sensor_ids[2]

            # Print results
            print(f"\n{nest['Nest_id']} ({nest['Species_code']}):")
            for i, (sensor_id, distance) in enumerate(zip(sensor_ids, distances), 1):
                print(f"  {i}. Sensor {sensor_id}: {distance:.3f} km")
        else:
            print(f"\n{nest['Nest_id']} ({nest['Species_code']}): Missing coordinates")

    return nests

def save_nest_locations(file_path, nests):
    """Save updated nest locations to CSV file."""
    fieldnames = ['Nest_id', 'lat', 'lon', 'Species_code', '1st_Closest', '2nd_closest', '3rd_closest']

    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nests)

def main():
    # File paths
    sensor_file = "data/sensor_locations.csv"
    nest_file = "data/nest_locations.csv"

    print("Loading sensor locations...")
    sensors = load_sensor_locations(sensor_file)
    print(f"Loaded {len(sensors)} sensors with valid coordinates")

    print("\nLoading nest locations...")
    nests = load_nest_locations(nest_file)
    print(f"Loaded {len(nests)} nests")

    print("\n" + "="*60)
    print("Finding nearest sensors for each nest...")
    print("="*60)

    updated_nests = update_nest_locations(nests, sensors)

    print("\n" + "="*60)
    print("Saving updated nest locations...")
    save_nest_locations(nest_file, updated_nests)
    print(f"Updated {nest_file} with nearest sensor information")
    print("="*60)

if __name__ == "__main__":
    main()
