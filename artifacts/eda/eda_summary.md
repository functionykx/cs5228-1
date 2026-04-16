# EDA Summary (train)

- n_train: 2666
- churn_rate: 0.1455

## Missing Values

- None

## Numeric Summary

```
                         count        mean        std   min      25%     50%      75%    max
Account length          2666.0  100.620405  39.563974   1.0   73.000  100.00  127.000  243.0
Number vmail messages   2666.0    8.021755  13.612277   0.0    0.000    0.00   19.000   50.0
Total day minutes       2666.0  179.481620  54.210350   0.0  143.400  179.95  215.900  350.8
Total day calls         2666.0  100.310203  19.988162   0.0   87.000  101.00  114.000  160.0
Total eve minutes       2666.0  200.386159  50.951515   0.0  165.300  200.90  235.100  363.7
Total eve calls         2666.0  100.023631  20.161445   0.0   87.000  100.00  114.000  170.0
Total night minutes     2666.0  201.168942  50.780323  43.7  166.925  201.15  236.475  395.0
Total night calls       2666.0  100.106152  19.418459  33.0   87.000  100.00  113.000  166.0
Total intl minutes      2666.0   10.237022   2.788349   0.0    8.500   10.20   12.100   20.0
Total intl calls        2666.0    4.467367   2.456195   0.0    3.000    4.00    6.000   20.0
Customer service calls  2666.0    1.562641   1.311236   0.0    1.000    1.00    2.000    9.0
```

## High Correlation Pairs (|corr| >= 0.95)

- None

## Top Features (Mutual Information)

- cat__International plan_Yes: 0.046600
- num__Total day minutes: 0.046425
- cat__International plan_No: 0.032397
- num__Customer service calls: 0.029483
- num__Number vmail messages: 0.018750
- cat__State_NM: 0.015167
- cat__State_NH: 0.015105
- cat__State_NJ: 0.012587
- cat__State_DE: 0.012316
- cat__State_AK: 0.010541
- num__Total intl minutes: 0.009615
- cat__State_WV: 0.007902
- num__Total day calls: 0.007883
- cat__State_NV: 0.007358
- cat__State_SC: 0.006832
- cat__State_MN: 0.006776
- num__Total intl calls: 0.006742
- cat__Voice mail plan_No: 0.006467
- cat__State_NY: 0.006033
- cat__Voice mail plan_Yes: 0.005347

## Plots

- cat_area_code_churn_rate.png
- cat_area_code_freq.png
- cat_international_plan_churn_rate.png
- cat_international_plan_freq.png
- cat_state_churn_rate.png
- cat_state_freq.png
- cat_voice_mail_plan_churn_rate.png
- cat_voice_mail_plan_freq.png
- correlation_heatmap.png
- num_account_length.png
- num_customer_service_calls.png
- num_number_vmail_messages.png
- num_total_day_calls.png
- num_total_day_minutes.png
- num_total_eve_calls.png
- num_total_eve_minutes.png
- num_total_intl_calls.png
- num_total_intl_minutes.png
- num_total_night_calls.png
- num_total_night_minutes.png
