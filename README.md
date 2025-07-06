# Zephyr

Zephyr is a smart route planner designed to help runners systematically explore their neighborhoods. It generates routes that prioritize new roads you haven't run yet, helping you discover more of your area while avoiding repetitive paths.

## Features

- **Smart Route Generation**: Multiple algorithms including zigzag patterns for efficient parallel street coverage
- **Progress Tracking**: Visual progress indicators showing completed roads and total mileage explored
- **Weather Integration**: Real-time weather data that adjusts route distances based on conditions
- **Preference Management**: Customizable temperature, wind, and distance preferences
- **Interactive Map**: Clean, modern interface with real-time weather display
- **Route Variety**: Four different route generation strategies to keep runs interesting

## How It Works

### Route Generation Algorithms

Zephyr uses four distinct route generation strategies, each optimized for different exploration patterns:

#### 1. Two-Leg Loops
Creates circular routes by finding an intermediate point and generating two paths: outbound and return. The algorithm:
- Samples multiple intermediate points at different distances from start
- Uses different weighting for outbound vs return legs to avoid repetition
- Filters routes based on target distance and new road percentage

#### 2. Multi-Point Routes
Generates routes with 3-4 waypoints for increased variety:
- Creates paths connecting multiple intermediate points
- Avoids repeating segments from previous legs
- Provides more complex route patterns

#### 3. Out-and-Back Routes
Traditional out-and-back patterns with different return paths:
- Finds turnaround points at optimal distances
- Uses different return routes to maximize new road coverage
- Simple but effective for systematic exploration

#### 4. Zigzag Routes (New)
Advanced algorithm for efficient parallel street coverage:
- Identifies streets with similar orientations (within 15-degree bins)
- Groups parallel streets and creates zigzag patterns between them
- Systematically covers multiple parallel streets in a single run
- Particularly effective in grid-like neighborhoods

### Technical Implementation

#### Subgraph Optimization
For performance, Zephyr creates a targeted subgraph around the starting point:
- Radius = 0.75 × target distance
- Reduces computation time from O(n²) to O(k²) where k << n
- Falls back to full graph if area is sparse (< 30 nodes)

#### Weighting System
Routes are scored using a sophisticated weighting function:
- **New Road Priority**: Undone roads get 0.8× weight, done roads get 15× penalty
- **Hill Avoidance**: Gradual penalty based on grade (1 + grade × 5)
- **Route Variety**: Strategy-specific bonuses (zigzag gets 1.2× multiplier)
- **Distance Optimization**: Closer to target distance gets higher score

#### Scoring Algorithm
Final route selection uses multi-factor scoring:
- Distance score: 1.0 / (1.0 + |actual - target| / target)
- Newness score: percentage_new / 100.0
- Strategy bonus: varies by route type
- Combined score: (distance × 0.4 + newness × 0.6) × strategy_bonus

### Weather Integration

Zephyr adjusts route distances based on current weather conditions:
- **Temperature**: Gradual reduction if outside ideal range (default 50-68°F)
- **Wind**: Distance reduction for high winds (default max 15 mph)
- **Adaptive Scaling**: Weather factor = min(temp_factor, wind_factor)
- **Minimum Distance**: Ensures routes never go below 1.0 miles

### Progress Tracking

The system tracks exploration progress using:
- **Edge-based tracking**: Each road segment marked as done/undone
- **Mileage calculation**: Total distance of completed vs available roads
- **Visual indicators**: Green for completed roads, gray for unexplored
- **Progress percentages**: Both edge count and mileage-based metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/conjfrnk/Zephyr.git
cd Zephyr
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser to `http://localhost:5000`

## Usage

1. **Load Your Area**: Enter zip codes for your neighborhood
2. **Set Preferences**: Configure ideal temperature, wind, and target distance
3. **Plan Routes**: Click "Plan New Route" and select a starting point on the map
4. **Choose Route**: Review multiple route options with different characteristics
5. **Complete Runs**: Mark routes as completed to track progress
6. **Explore More**: The system will suggest routes covering new territory

## Technical Architecture

### Backend (Flask + SQLAlchemy)
- **Graph Processing**: NetworkX for route algorithms
- **Spatial Data**: OSMnx for OpenStreetMap integration
- **Weather API**: National Weather Service integration
- **Database**: SQLite with SQLAlchemy ORM

### Frontend (Vanilla JavaScript + Leaflet)
- **Map Interface**: Leaflet.js for interactive mapping
- **Real-time Updates**: WebSocket-like polling for progress updates
- **Responsive Design**: Modern CSS with card-based layout
- **Contextual Feedback**: Inline message containers for user feedback

### Data Models
- **Run**: Stores planned and completed routes
- **DoneEdge**: Tracks completed road segments
- **Pref**: User preferences and settings

## Performance Optimizations

- **Subgraph Creation**: Reduces computation time by 80-90%
- **Caching**: Weather data cached with LRU cache
- **Async Processing**: Graph loading in background threads
- **Smart Sampling**: Limits intermediate points to prevent exponential growth

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.