# GalaxyPPG
![](https://img.shields.io/badge/contact-dataset@kse.kaist.ac.kr-red)
![](https://img.shields.io/badge/doi-https://XXXX.XXXX.XXX-blue)

_Collected by Interactive Computing Lab at KAIST_

## Motivation
GalaxyPPG was developed to determine whether HR/HRV features can be accurately extracted using **Photoplethysmography (PPG)** sensor data, one of the key data types collected from Samsung's Galaxy Watch series, for physiological and psychological stress detection. To assess accuracy, a comparison was made against the Polar H10, which is considered the gold standard in HR/HRV measurement with its Electrocardiography (ECG) technology, as well as the Empatica E4, another wrist-worn PPG sensing device widely used by research communities.

## Data Format and Structure
### File tree
The overall folder structure is as follows:
```plaintext
GalaxyPPG/
└─ Dataset/
   ├─ Meta.csv
   ├─ P01/
   ├─ P02/
   |  ├─ E4/
   |  |  ├─ ACC.csv
   |  |  ├─ BVP.csv
   |  |  ├─ HR.csv
   |  |  ├─ IBI.csv
   |  |  ├─ TEMP.csv
   |  ├─ GalaxyWatch/
   |  |  ├─ ACC.csv
   |  |  ├─ HR.csv
   |  |  ├─ PPG.csv
   |  |  ├─ SkinTemp.csv*
   |  ├─ PolarH10/
   |  |  ├─ ACC.csv
   |  |  ├─ ECG.csv
   |  |  ├─ HR.csv
   |  |  ├─ IBI.csv
   |  └─ Event.csv
   ├─ ...
   └─ P24/
```

The data for each participant is stored in individual subfolders named according to participant IDs (e.g., P01, P02, up to P24). Within each participant’s folder, the data is further organized by the devices used to collect physiological signals: Empatica E4, Galaxy Watch 5, and Polar H10.

⚠ Note for missing data:
* P01's Galaxy Watch data and event annotation is not recorded.
* P07 and P08's HR measurement of Galaxy Watch was not recorded.
* P02's skin temperature of Galaxy Watch was not recorded.

### File Description

* `Meta.csv`: Metadata about participant and collection.
    | **Column**  | **Description** |
    | -- | -- |
    | UID (string) | Particpant's ID |
    | AGE (int) | Age of participant |
    | GENDER (string) | Gender of participant (F: Female, M: Male) |
    | TSST (int) | percieved stress report as 7-point Likert scale for TSST (lower number means less stress) |
    | SSST (int) | percieved stress report as 7-point Likert scale for SSST (lower number means less stress) |
    | GalaxyWatch (string) | The wrist-position of GalaxyWatch (L: left, R: right) |
    
* `[UID]/Event.csv`: The log of activity changes during the collection. 
    | **Column**  | **Description** |
    | -- | -- |
    | timestamp (long, ms) | The Unix timestamp in milliseconds when the event has changed |
    | session (string) | The change of the activity |
    | status (string) | The change status of the activity:  <br/> <ul><li>**ENTER**: Indicating activity's start</li><li>**EXIT**: Indicating activity's finish</li></ul>|

* `[UID]/GalaxyWatch/ACC.csv`: Accerelation signal recorded at 25HZ. 
    | **Column**  | **Description** |
    | -- | -- |
    | dataReceived (long, ms) | The Unix timestamp in milliseconds when the data is received by the logger application |
    | timestamp (long, ms) | The Unix timestamp in milliseconds when the data is recorded |
    | x (float, m/s²) | The X-axis acceleration data |
    | y (float, m/s²) | The Y-axis acceleration data |
    | z (float, m/s²) | The Z-axis acceleration data |
    
* `[UID]/GalaxyWatch/HR.csv`: Heart rate (HR) and Inter-beat-interval (IBI) data processed at 1HZ by Galaxy Watch.
    | **Column**  | **Description** |
    | -- | -- |
    | dataReceived (long, ms) | The Unix timestamp in milliseconds when the data is received by the logger application |
    | timestamp (long, ms) | The Unix timestamp in milliseconds when the data is recorded |
    | hr (int, bpm) | The heart rate |
    | hrStatus (int) | The status code provided by Galaxy Watch. The meaning of each number is as follows: <br/> <ul><li>**-999**: A higher priority sensor is operating (e.g., BIA).</li><li>**-99**: `flush()` is called but no data.</li><li>**-10**: PPG signal is too weak or the user's movement is excessive.</li><li>**-8**: PPG signal is weak or there is some user movement.</li><li>**-3**: Wearable is detached.</li><li>**-2**: Wearable movement is detected.</li><li>**0**: Initial heart rate measuring state or a higher priority sensor is operating (e.g., BIA).</li><li>**1**: Successful heart rate measurement.</li></ul> ⚠ However, it is better to note that the status code as -11 and -1 were also recorded. |
    | ibi (array of int, ms) | The IBI provided as a list. Since the GalaxyWatch used batching for providing data, ibi was retreived with first data point of HR  |
    | ibiStatus (array of int) | The status code provided by GalaxyWatch for each IBI value:<br> <ul><li><strong>-1</strong>: Error</li><li><strong>0</strong>: Normal</li></ul> |

* `[UID]/GalaxyWatch/PPG.csv`: The PPG signal sampling at 25HZ.
    | **Column**  | **Description** |
    | -- | -- |
    | dataReceived (long, ms) | The Unix timestamp in milliseconds when the data is received by the logger application |
    | timestamp (long, ms) | The Unix timestamp in milliseconds when the data is recorded |
    | ppg (long) | The PPG green LED's ADC (analog-to-digital) value |
    | status (int) | The status code of ppg<br/> <ul><li>**-999**: A higher priority sensor is operating (e.g., BIA).</li><li>**0**: Normal value.</li><li>**500**: The `STATUS` interface is not supported due to outdated watch software.</li></ul> |

* `[UID]/GalaxyWatch/SkinTemp.csv`: The skin temperature sampling for each minute (1/60HZ).
    | **Column**  | **Description** |
    | -- | -- |
    | dataReceived (long, ms) | The Unix timestamp in milliseconds when the data is received by the logger application |
    | timestamp (long, ms) | The Unix timestamp in milliseconds when the data is recorded |
    | ambientTemp (float, °C) | An ambient temperature around the Galaxy Watch |
    | objectTemp (float, °C) | A object's temperature contacted with the Galaxy Watch |
    | status (int) | The status code of skin temperature:<br><ul><li>**-1**: Error</li><li>**0**: Normal</li></ul>|

* `[UID]/E4/ACC.csv`: The accerelation signal recorded at 32HZ.
    | **Column**  | **Description** |
    | -- | -- |
    | timestamp (long, μs) | The Unix timestamp in microeconds when the data is recorded |
    | x (float, 1/64g) | The X-axis acceleration data |
    | y (float, 1/64g) | The X-axis acceleration data |
    | z (float, 1/64g) | The X-axis acceleration data |

* `[UID]/E4/BVP.csv`: The data derived from PPG for blood volume pressure (BVP).
    | **Column**  | **Description** |
    | -- | -- |
    | timestamp (long, μs) | The Unix timestamp in microeconds when the data is recorded |
    | value (float) | The relative value that represent the change of blood flow |

* `[UID]/E4/TEMP.csv`: The temperature
    | **Column**  | **Description** |
    | -- | -- |
    | timestamp (long, μs) | The Unix timestamp in microeconds when the data is recorded |
    | value (float, °C) | The relative value that represent the change of blood flow |

* `[UID]/E4/HR.csv`: The average heart rate processed at 1HZ.
    | **Column**  | **Description** |
    | -- | -- |
    | timestamp (long, μs) | The Unix timestamp in microeconds when the data is recorded |
    | value (float, bpm) | The average heart rate |

* `[UID]/E4/IBI.csv`: The IBI data.
    | **Column**  | **Description** |
    | -- | -- |
    | timestamp (long, μs) | The Unix timestamp in microeconds when the data is recorded |
    | duration (long, μs) | The ibi value |

* `[UID]/PolarH10/ACC.csv`: The acceleration signal recorded at 200HZ.
    | **Column**  | **Description** |
    | -- | -- |
    | phoneTimestamp (long, ms)  | the timestamp in milliseconds in UTC+9 |
    | sensorTimestamp (long, ns) | the timestamp expressed in nanoseconds for sensor |
    | x (int, mg) | The X-axis acceleration data |
    | y (int, mg) | The X-axis acceleration data |
    | z (int, mg) | The X-axis acceleration data |

* `[UID]/PolarH10/ECG.csv`: The acceleration signal recorded at 130HZ.
    | **Column**  | **Description** |
    | -- | -- |
    | phoneTimestamp (long, ms)  | the timestamp in milliseconds in UTC+9 |
    | sensorTimestamp (long, ns) | the timestamp expressed in nanoseconds for sensor |
    | ecg (int, μV) | The ECG value |

* `[UID]/PolarH10/HR.csv`: The heartrate processed at 1HZ.
    | **Column**  | **Description** |
    | -- | -- |
    | phoneTimestamp (long, ms)  | the timestamp in milliseconds in UTC+9 |
    | hr (int, bpm) | The heart rate |
    | hrv (float) | Unknown. The data was provided by H10, but the exact meaning is unknown |

* `[UID]/PolarH10/IBI.csv`: The processed IBI.
    | **Column**  | **Description** |
    | -- | -- |
    | phoneTimestamp (long, ms)  | the timestamp in milliseconds in UTC+9 |
    | duration (int, ms) | The ibi value |

## Data Logging Application
1. **Polar H10**:  
    * Data was collected by connecting the device to a smartphone using the [Polar Sensor Logger](https://play.google.com/store/apps/details?id=com.j_ware.polarsensorlogger&hl=en) app.
    * Data could be saved directly to the smartphone.

2. **Galaxy Watch 5**:  
    * Data was collected by connecting the device to a smartphone using a custom-developed [Galaxy Watch Logger application](https://github.com/Kaist-ICLab/GalaxyPPG-Logger).
    * Data could be saved directly to the smartphone.

3. **Empatica E4**:  
    * Data was collected using the [E4 Realtime](https://play.google.com/store/apps/details?id=com.empatica.e4realtime&hl=en) app provided by Empatica, connected to a smartphone.
    * The data could be downloaded from the E4 cloud.


