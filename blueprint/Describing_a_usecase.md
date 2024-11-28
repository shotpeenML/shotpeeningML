## Input Shot Peening Recipe and Geometry
Description: The user enters the geometry (3D model, dimensions, or relevant descriptors) and shot peening recipe (intensity, duration, shot type, angle, etc.).
Actors: Mechanical Engineer, Process Engineer
Outcome: The system prepares the input data for the ML model to predict deformation.

## Predict and Visualize Deformation
Description: Based on the input geometry and recipe, the ML model predicts deformation, and the result is visualized on the 3D model.
Actors: Mechanical Engineer, Process Engineer
Outcome: Predicted deformation is displayed, allowing the user to assess the impact of the shot peening process.

## Compare Recipes for Optimal Deformation
Description: The user selects multiple recipes, and the system compares predicted deformations for each, highlighting optimal options.
Actors: Process Engineer
Outcome: The user can determine the best shot peening settings for desired deformation characteristics.

## Store and Retrieve Shot Peening Recipes and Results
Description: The user can save a shot peening recipe and its corresponding predicted deformation, making it accessible for future reference.
Actors: Process Engineer, Quality Control Engineer
Outcome: Users have a repository of previous recipes and results, useful for process refinement and future predictions.

## Adjust Parameters to Test Deformation Impact
Description: The user can iteratively adjust parameters (e.g., intensity, angle, shot size) and see how these changes affect the deformation prediction.
Actors: Mechanical Engineer, Process Engineer
Outcome: Users can experiment with various settings to optimize the shot peening process without physical trials.
