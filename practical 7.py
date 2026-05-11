import json

# Step 1: Sample JSON data

data = [
 {"name": "Alice", "age": 25, "city": "Mumbai"},
 {"name": "Bob", "age": 30, "city": "Delhi"},
 {"name": "Charlie", "age": 35, "city": "Mumbai"}
]

print("Original Data:")
print(json.dumps(data, indent=2))

# Step 2: Filter

filtered = [p for p in data if p["city"] == "Mumbai"]

print("\nFiltered Data:")
print(json.dumps(filtered, indent=2))

# Step 3: Transform

names = [p["name"] for p in filtered]

print("\nNames from filtered data:")
print(names)

# Step 4: Group

grouped = {}

for p in data:
    city = p["city"]

    if city not in grouped:
        grouped[city] = []

    grouped[city].append(p)

print("\nGrouped Data by City:")
print(json.dumps(grouped, indent=2))

# Step 5: Aggregate

avg_age = sum(p["age"] for p in filtered) / len(filtered)

print("\nAverage Age in Mumbai:")
print(avg_age)
