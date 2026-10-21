# Day 1 Lab: Answers

Name: Abdullah Akram Gondal

## 1. Leakage
Which columns did you drop and why? Is `tsunami` leaky? Is `magType`?

I dropped title, sig, mmi, cdi, felt, alert and tsunami. title contains the magnitude in its text and sig is calculated from the magnitude, so both give away the answer. mmi, cdi, felt and alert are filled only after people feel the quake, so we do not know them at prediction time. tsunami is also set after the event, so I treat it as leaky. magType is not leaky because it only says how the magnitude was measured (ml, mb, mw), not what it is, so I kept it.


## 2. Stream vs batch
How many events changed (same `id`, newer `updated`) during your stream window? What does that tell you about "latest version wins"?

In my short stream run (about 30 seconds), 13 events were collected. After merging them with the weekly data, 8 events had the same id but a newer updated time. This means USGS changes events after the first report. So we keep only the latest version of each event, otherwise we would use old and wrong values.

## 3. Outliers
Your decision on negative depth and negative magnitude, with reasoning.

Outliers: I ran iqr_outlier_mask on depth_km and mag. Out of 1814 earthquakes, it flagged 270 depth outliers and 183 magnitude outliers. 71 rows have negative depth (lowest is -3.4) and 88 rows have negative magnitude (lowest is -1.22). I think both are real measurements, not errors. Depth is measured from sea level, so a shallow quake in a high area can have a negative depth, and magnitude is on a log scale, so very small quakes can go below zero. The deep values (up to 637 km) are real deep earthquakes. I kept all these rows because deleting them would remove real small and deep events, and I will use a robust scaler later so the extreme values do not dominate.


## 4. Cardinality
You grouped `region` to top-k. Name one alternative encoding and one risk it carries.

One other way is target encoding: we replace each region with the average big_quake rate of that region. The risk is data leakage, because the encoding uses the target. If we calculate it before the split, the model sees the answer. So it must be fitted on the train set only.

