# Solow Growth Model - Differential Equations Project

## Overview
This project explores the Solow-Cobb Douglas growth model using a differential equations approach. We replicate and build upon the methodology from a research paper that applied the Extended Kalman Filter (EKF) to dynamically track changes in economic growth parameters. Our goal was to simplify this complex model into an autonomous differential equation that could be visualized through slope fields, offering a clearer understanding of capital evolution over time.

## Problem Motivation
Ghana's economy, like many developing economies, faces challenges in forecasting future growth due to limited and variable historical data. To address this, we studied the U.S. economy, which offers rich historical data, and applied modern filtering techniques to better understand dynamic changes in economic variables like productivity, capital, and labor.

## Methods
- **Data Gathering:** We collected U.S. economic data for capital, labor, saving rates, productivity factors, labor growth rates, and depreciation rates from the Federal Reserve Economic Database (FRED).
- **Model Simplification:** To create a slope field, we "froze" parameter values based on recent data (2015-2019) where the productivity factor was indexed to 1.
- **Equation Used:**
  \[ \frac{dk}{dt} = s A k^{1-\alpha} - (n + \delta)k \]
  where:
  - **k** = Capital per effective worker
  - **s** = Saving rate
  - **A** = Productivity factor
  - **\alpha** = Capital's share of output
  - **n** = Labor force growth rate
  - **\delta** = Depreciation rate

- **Initial Condition:** We selected initial conditions around the year when the productivity factor equals 1 (2017), to align with our frozen parameters.

## Parameters (Frozen Values)
- Saving rate (**s**): 6.36%
- Productivity factor (**A**): 1
- Output elasticity of capital (**\alpha**): 0.3
- Labor growth rate (**n**): 0.0726
- Depreciation rate (**\delta**): 2276.684

## Key Results
- **Slope Field Behavior:** We expected the slope field to show steep declines in capital due to the very large depreciation rate.
- **Simplified Analysis:** Freezing the parameters gave a snapshot of the model's behavior without real-time parameter variation.

## Future Work
- Incorporate Unscented Kalman Filter (UKF) to improve real-time tracking.
- Adapt the model for Ghana-specific data as more becomes available.
- Introduce parameter sensitivity studies to see how small changes affect the growth path.

## Credits
- Research foundation based on the paper "Application of Extended Kalman Filtering to the Solow-Cobb Douglas Growth Model."
- Data sourced from FRED Economic Data.

## License
This project is licensed under the MIT License.

