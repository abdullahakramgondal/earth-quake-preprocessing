# Day 1 Lab: Answers

Name:

## 1. Leakage
Which columns did you drop and why? Is `tsunami` leaky? Is `magType`?


## 2. Stream vs batch
How many events changed (same `id`, newer `updated`) during your stream window? What does that tell you about "latest version wins"?


## 3. Outliers
Your decision on negative depth and negative magnitude, with reasoning.
Outliers: I ran iqr_outlier_mask on depth_km and mag. Out of 1814 earthquakes, it flagged 270 depth outliers and 183 magnitude outliers. 71 rows have negative depth (lowest is -3.4) and 88 rows have negative magnitude (lowest is -1.22). I think both are real measurements, not errors. Depth is measured from sea level, so a shallow quake in a high area can have a negative depth, and magnitude is on a log scale, so very small quakes can go below zero. The deep values (up to 637 km) are real deep earthquakes. I kept all these rows because deleting them would remove real small and deep events, and I will use a robust scaler later so the extreme values do not dominate.


## 4. Cardinality
You grouped `region` to top-k. Name one alternative encoding and one risk it carries.

