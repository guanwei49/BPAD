# Survey and Benchmark of Anomaly Detection in Business Processes

This is the source code of our paper '[Survey and Benchmark of Anomaly Detection in Business Processes](https://ieeexplore.ieee.org/abstract/document/10738505)'.

**Abstract**: Effective management of business processes is crucial for organizational success. However, despite meticulous design and implementation, anomalies are inevitable and can result in inefficiencies, delays, or even significant financial losses.
Numerous methods for detecting anomalies in business processes have been proposed recently. However, there is no comprehensive benchmark to evaluate these methods. Consequently, the relative merits of each method remain unclear due to differences in their experimental setup, choice of datasets and evaluation measures.
In this paper, we present a systematic literature review and taxonomy of business process anomaly detection methods. Additionally,  we select at least one method from each category, resulting in 16 methods that are cross-benchmarked against 32 synthetic logs and 19 real-life logs from different industry domains.
Our analysis provides insights into the strengths and weaknesses of different anomaly detection methods. Ultimately, our findings can help researchers and practitioners in the field of process mining make informed decisions when selecting and applying anomaly detection methods to real-life business scenarios.
Finally, some future directions are discussed in order to promote the evolution of business process anomaly detection.

## Appendix Tables
![A1.png](pic/A1.png)
![A2.png](pic/A2.png)
![A3.png](pic/A3.png)
![A4.png](pic/A4.png)
![A5.png](pic/A5.png)

## Studied Models

### Fourteen unsupervised methods

|   Model    | Year |    Perspective    |   Granularity   |                                                                                                                                Paper                                                                                                                                |
|:----------:|:----:|:-----------------:|:---------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   Naive    | 2013 |   Control-flow    |   Trace-level   |                                                 [Algorithms for Anomaly Detection of Traces in Logs of Process Aware Information Systems](https://www.sciencedirect.com/science/article/abs/pii/S0306437912000567)                                                  |
|  Sampling  | 2013 |   Control-flow    |   Trace-level   |                                                 [Algorithms for Anomaly Detection of Traces in Logs of Process Aware Information Systems](https://www.sciencedirect.com/science/article/abs/pii/S0306437912000567)                                                  |
| Likelihood | 2016 | Multi-perspective | Attribute-level |                                                              [Multi-perspective Anomaly Detection in Business Process Execution Events](https://link.springer.com/chapter/10.1007/978-3-319-48472-3_5)                                                              |
|    DAE     | 2018 | Multi-perspective | Attribute-level |                                                                       [Analyzing Business Process Anomalies Using Autoencoders](https://link.springer.com/article/10.1007/s10994-018-5702-8)                                                                        |
|  BINetv2   | 2018 | Multi-perspective | Attribute-level |                                                            [BINet: Multivariate Business Process Anomaly Detection Using Deep Learning](https://link.springer.com/chapter/10.1007/978-3-319-98648-7_16)                                                             |
|    VAE     | 2019 | Multi-perspective | Attribute-level |                                                                 [Autoencoders for Improving Quality of Process Event Logs](https://www.sciencedirect.com/science/article/abs/pii/S0957417419302829)                                                                 |
|    LAE     | 2019 | Multi-perspective | Attribute-level |                                                                 [Autoencoders for Improving Quality of Process Event Logs](https://www.sciencedirect.com/science/article/abs/pii/S0957417419302829)                                                                 |
|  W2V-LOF   | 2020 |   Control-flow    |   Trace-level   |                                                                              [Anomaly Detection on Event Logs with a Scarcity of Labels](https://ieeexplore.ieee.org/document/9230308)                                                                              |
| VAE-OCSVM  | 2021 |   Control-flow    |   Trace-level   |             [Variational Autoencoder for Anomaly Detection in Event Data in Online Process Mining](https://www.semanticscholar.org/paper/Variational-Autoencoder-for-Anomaly-Detection-in-in-Krajsic-Franczyk/010abb71527b0a1f6b38eb83c7503d00a67cb49a)             |
|    GAE     | 2021 | Multi-perspective |   Trace-level   |                                                                     [Graph Autoencoders for Business Process Anomaly Detection](https://link.springer.com/chapter/10.1007/978-3-030-85469-0_26)                                                                     |
|  Leverage  | 2022 |   Control-flow    |   Trace-level   |                                      [Keeping Our Rivers Clean: Information-Theoretic Online Anomaly Detection for Streaming Business Process Events](https://www.sciencedirect.com/science/article/abs/pii/S0306437921001125)                                      |
|  BINetv3   | 2022 | Multi-perspective | Attribute-level |                                                             [BINet: Multi-perspective Business Process Anomaly Classification](https://www.sciencedirect.com/science/article/abs/pii/S0306437919305101)                                                             |
|  GRASPED   | 2023 | Multi-perspective | Attribute-level |                                                             [GRASPED: A GRU-AE Network Based Multi-Perspective Business Process Anomaly Detection Model](https://ieeexplore.ieee.org/document/10088425)                                                             |
|    GAMA    | 2024 | Multi-perspective | Attribute-level | [GAMA: A Multi-graph-based Anomaly Detection Framework for Business Processes via Graph Neural Networks](https://www.techrxiv.org/articles/preprint/GAMA_A_Multi-graph-based_Anomaly_Detection_Framework_for_Business_Processes_via_Graph_Neural_Networks/23627850) |

### Two weakly-supervised methods

| Model | Year |    Perspective    |   Granularity   |                                                                                Paper                                                                                |
|:-----:|:----:|:-----------------:|:---------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  DRL  | 2023 |   Control-flow    |   Trace-level   | [Deep Reinforcement Learning for Data-efficient Weakly Supervised Business Process Anomaly Detection](https://link.springer.com/article/10.1186/s40537-023-00708-5) |
| WAKE  | 2024 | Multi-perspective | Attribute-level |   [WAKE: A Weakly Supervised Business Process Anomaly Detection Framework via a Pre-Trained Autoencoder](https://ieeexplore.ieee.org/abstract/document/10285076)    |


**In general, 10% of anomalies are labeled during training in these weakly-supervised methods.**

## Requirements

### Win

- [PyTorch==1.13.0](https://pytorch.org)
- [tensorflow-gpu==2.6.0](https://www.tensorflow.org/)
- [NumPy==1.21.5](https://numpy.org)
- [scikit-learn==1.0.2](https://scikit-learn.org)
- [pm4py==2.7.4](https://pm4py.fit.fraunhofer.de/)
- [pyg==2.3.0](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [pandas==1.3.5](https://pandas.pydata.org/)
- [gensim==4.1.2](https://radimrehurek.com/gensim/)

### Linux

- [PyTorch==1.13.0](https://pytorch.org)
- [tensorflow==2.10.0](https://www.tensorflow.org/)
- [NumPy==1.24.3](https://numpy.org)
- [scikit-learn==1.3.0](https://scikit-learn.org)
- [pm4py==2.7.4](https://pm4py.fit.fraunhofer.de/)
- [pyg==2.3.0](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [pandas==2.0.3](https://pandas.pydata.org/)
- [gensim==4.3.0](https://radimrehurek.com/gensim/)

## Examples

    ```
    python \generator\gen_anomalous_eventlog_syn.py  # Generate synthetic datasets (event logs) containing anomalies and save them into the 'eventlogs' directory.
    python \generator\gen_anomalous_real_life_log.py  # Generate real-life datasets (event logs) containing anomalies and save them into the 'eventlogs' directory.
    python main_unsup.py # get the result for each unsupervised method.
    python main_weakly.py # get the result for each weakly-supervised method.
    ```

## Datasets

Nine commonly used real-life datasets:

i) **_[BPIC12](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)_**: Event log of a loan application
process

ii) **_[BPIC13](https://doi.org/10.4121/uuid:a7ce5c55-03a7-4583-b855-98b86e1a2b07)_**: Logs of Volvo IT incident and
problem management.

iii) **_[BPIC15](https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1)_**: This data is provided by five
Dutch municipalities. The data contains all building permit
applications over a period of approximately four years.

iv) **_[BPIC17](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b)_**: This event log pertains to a loan
application process of a Dutch financial institute. The data
contains all applications filed through an online system in 2016 and their subsequent events until February 1st 2017,
15:11.

v) **_[BPIC20](https://doi.org/10.4121/uuid:52fb97d4-4588-43c9-9d04-3604d4613b51)_**: The dataset contains events
pertaining to two years of travel expense claims. In 2017, events were
collected for two departments, in 2018 for the entire university.

vi) **_[Billing](https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741)_**: This log contains events that
pertain to the billing of medical services provided by a hospital.

vii) **_[Receipt](https://doi.org/10.4121/12709127.v2)_**: This log contains records of the receiving phase of the
building permit application process in an
anonymous municipality.

viii) **_[RTFMP](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5)_**: Real-life event log of an
information system managing road traffic fines.

ix) **_[Sepsis](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460)_**: This log contains events of
sepsis cases from a hospital.

Eight synthetic logs: i.e., **_Paper_**,  _**P2P**_, **_Small_**, **_Medium_**, **_Large_**, **_Huge_**, **_Gigantic_**,
and **_Wide_**.

The summary of statistics for each event log (without anomalies) is presented below:

|    Log     | #Activities | #Traces |   #Events   | Max trace length | Min trace length | #Attributes | #Attribute values |
|:----------:|:-----------:|:-------:|:-----------:|:----------------:|:----------------:|:-----------:|:-----------------:|
|  Gigantic  |    76-78    |  5000   | 28243-31989 |        11        |        3         |     1-4     |      70-363       |
|    Huge    |     54      |  5000   | 36377-42999 |        11        |        5         |     1-4     |      69-340       |
|   Large    |     42      |  5000   | 51099-56850 |        12        |        10        |     1-4     |      68-292       |
|   Medium   |     32      |  5000   | 28416-31372 |        8         |        3         |     1-4     |      66-276       |
|    P2P     |     13      |  5000   | 37941-42634 |        11        |        7         |     1-4     |      39-146       |
|   Paper    |     14      |  5000   | 49839-54390 |        12        |        9         |     1-4     |      36-128       |
|   Small    |     20      |  5000   | 42845-46060 |        10        |        7         |     1-4     |      39-144       |
|    Wide    |    23-34    |  5000   | 29128-31228 |        7         |       5-6        |     1-4     |      53-264       |
|   BPIC12   |     36      |  13087  |   262200    |       175        |        3         |      0      |         0         |
| BPIC13\_C  |      7      |  1487   |    6660     |        35        |        1         |      4      |        638        |
| BPIC13\_I  |     13      |  7554   |    65533    |       123        |        1         |      4      |       2144        |
| BPIC13\_O  |      5      |   819   |    2351     |        22        |        1         |      2      |        251        |
| BPIC15\_1  |     398     |  1199   |    52217    |       101        |        2         |      2      |        49         |
| BPIC15\_2  |     410     |   832   |    44354    |       132        |        1         |      2      |        20         |
| BPIC15\_3  |     383     |  1409   |    59681    |       124        |        3         |      3      |        419        |
| BPIC15\_4  |     356     |  1053   |    47293    |       116        |        1         |      2      |        22         |
| BPIC15\_5  |     389     |  1156   |    59083    |       154        |        5         |      2      |        38         |
|   BPIC17   |     26      |  31509  |   1202267   |       180        |        10        |      1      |        149        |
| BPIC20\_D  |     17      |  10500  |    56437    |        24        |        1         |      2      |         9         |
| BPIC20\_I  |     34      |  6449   |    72151    |        27        |        3         |      2      |        10         |
| BPIC20\_PE |     51      |  7065   |    86581    |        90        |        3         |      2      |        10         |
| BPIC20\_PR |     29      |  2099   |    18246    |        21        |        1         |      2      |        10         |
| BPIC20\_R  |     19      |  6886   |    36796    |        20        |        1         |      2      |        10         |
|  Billing   |     18      | 100000  |   451359    |       217        |        1         |      0      |         0         |
|  Receipt   |     27      |  1434   |    8577     |        25        |        1         |      2      |        58         |
|   RTFMP    |     11      | 150370  |   561470    |        20        |        2         |      0      |         0         |
|   Sepsis   |     16      |  1050   |    15214    |       185        |        3         |      1      |        26         |

To simulate anomalous behavior, we inject artificial anomalies of six different types, with a proportion of $\rho$ percent (i.e., $\rho$ percent of the traces are anomalous)  into both synthetic and real-life logs.  
In our experiments, we assess the scalability of methods by considering values of $\rho$ at 5, 10, 15, 20, 25, 30, 35, 40, and 45.
To provide clarity, we define these anomaly types as follows:
![anomaly_type.png](pic/anomaly_type.png)
  - _Skip_: Bypassing a series of up to two events.
  - _Insert_: Interjecting a series of up to two random events.
  - _Rework_: Repeatedly carrying out a series of up to three events.
  - _Early_:  Executing up to two events prematurely, causing them to be bypassed at their subsequent designated point.
  - _Late_: Postponing the execution of up to two events, causing them to be bypassed at their earlier designated point.
  - _Attribute_: In up to two certain events, substituting the correct values of certain attributes with erroneous ones.



## Experiment Results

Critical difference diagram over trace-level anomaly detection (_F-score_) :
![model](pic/multiType_trace_f1.png)

Critical difference diagram over attribute-level anomaly detection (_F-score_ ):
![model](pic/multiType_attr_f1.png)

Critical difference diagram over trace-level anomaly detection (_AP_) :
![model](pic/multiType_trace_aupr.png)

Critical difference diagram over attribute-level anomaly detection (_AP_):
![model](pic/multiType_attr_aupr.png)

_F-score_ of trace-level anomaly detection:

|            | Gigantic  |   Huge    |   Large   |  Medium   |    P2P    |   Paper   |   Small   |   Wide    |  BPIC12   | BPIC13\_C | BPIC13\_I | BPIC13\_O | BPIC15\_1 | BPIC15\_2 | BPIC15\_3 | BPIC15\_4 | BPIC15\_5 |  BPIC17   | BPIC20\_D | BPIC20\_I | BPIC20\_PE | BPIC20\_PR | BPIC20\_R |  Billing  |  Receipt  |   RTFMP   |  Sepsis   |
|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:----------:|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|   Naive    |   0.726   |   0.884   |   0.907   |   0.840   |   0.901   |   0.908   |   0.909   |   0.901   |   0.451   |   0.474   |   0.464   |   0.417   |   0.396   |   0.384   |   0.395   |   0.385   |   0.383   |   0.484   |   0.803   |   0.696   |   0.620    |   0.727    |   0.797   |   0.789   |   0.722   |   0.882   |   0.414   |
|  Sampling  |   0.819   |   0.892   |   0.909   |   0.859   |   0.889   |   0.901   |   0.897   |   0.890   |   0.413   |   0.454   |   0.428   |   0.390   |   0.390   |   0.383   |   0.387   |   0.384   |   0.383   |   0.394   |   0.737   |   0.506   |   0.456    |   0.642    |   0.746   |   0.701   |   0.664   |   0.724   |   0.386   |
|  Leverage  |   0.763   |   0.895   |   0.914   |   0.868   |   0.901   |   0.908   |   0.908   |   0.898   |   0.512   |   0.539   |   0.525   |   0.446   |   0.000   |   0.000   |   0.000   |   0.000   |   0.000   |   0.411   |   0.816   |   0.691   |   0.580    |   0.727    |   0.805   |   0.476   | **0.755** |   0.916   |   0.417   |
| Likelihood |   0.739   |   0.875   |   0.894   |   0.841   |   0.941   |   0.950   |   0.967   |   0.869   |   0.773   |   0.383   |   0.420   |   0.322   |   0.445   |   0.426   |   0.459   |   0.452   |   0.453   |   0.512   |   0.902   |   0.826   |   0.741    | **0.813**  |   0.884   |   0.867   |   0.594   |   0.906   | **0.568** |
|  W2V-LOF   |   0.668   |   0.804   |   0.833   |   0.762   |   0.796   |   0.846   |   0.829   |   0.808   |   0.461   |   0.454   |   0.457   |   0.392   |   0.400   |   0.400   |   0.431   |   0.415   |   0.409   |   0.475   |   0.581   |   0.615   |   0.571    |   0.649    |   0.589   |   0.395   |   0.694   |   0.343   |   0.433   |
| VAE-OCSVM  |   0.385   |   0.378   |   0.383   |   0.382   |   0.379   |   0.381   |   0.382   |   0.379   |   0.331   |   0.335   |   0.383   |   0.271   |   0.393   |   0.386   |   0.389   |   0.391   |   0.385   |   0.383   |   0.379   |   0.385   |   0.385    |   0.386    |   0.378   |   0.254   |   0.377   |   0.257   |   0.380   |
|    GAE     |   0.509   |   0.475   |   0.593   |   0.506   |   0.614   |   0.535   |   0.526   |   0.641   |   0.411   |   0.406   |   0.394   |   0.343   |   0.392   |   0.386   |   0.392   |   0.390   |   0.386   |   0.383   |   0.554   |   0.420   |   0.405    |   0.463    |   0.549   |   0.381   |   0.413   |   0.436   |   0.387   |
|    DAE     |   0.751   |   0.804   |   0.836   |   0.848   |   0.867   |   0.892   |   0.910   |   0.876   |   0.585   |   0.443   |   0.469   |   0.444   |   0.402   |   0.399   |   0.403   |   0.403   |   0.401   |   0.414   |   0.873   |   0.586   |   0.558    |   0.596    |   0.850   |   0.731   |   0.648   |   0.861   |   0.495   |
|    VAE     |   0.571   |   0.625   |   0.749   |   0.698   |   0.667   |   0.820   |   0.781   |   0.725   |   0.576   |   0.417   |   0.412   |   0.443   |   0.401   |   0.396   |   0.405   |   0.403   |   0.397   |   0.404   |   0.663   |   0.488   |   0.466    |   0.486    |   0.604   |   0.593   |   0.615   |   0.845   |   0.491   |
|    LAE     |   0.463   |   0.501   |   0.511   |   0.542   |   0.586   |   0.618   |   0.596   |   0.534   |   0.862   |   0.402   |   0.400   | **0.463** |   0.403   |   0.416   |   0.400   |   0.423   |   0.398   |   0.386   |   0.880   |   0.702   |   0.613    |   0.666    |   0.847   |   0.798   |   0.485   |   0.834   |   0.519   |
|  BINetv2   |   0.511   |   0.552   |   0.541   |   0.534   |   0.581   |   0.557   |   0.563   |   0.527   |   0.709   |   0.334   | **0.582** |   0.282   |   0.399   |   0.385   |   0.396   |   0.388   |   0.393   |   0.667   |   0.843   |   0.775   |   0.769    |   0.787    |   0.835   |   0.710   |   0.441   |   0.762   |   0.396   |
|  BINetv3   |   0.531   |   0.538   |   0.564   |   0.572   |   0.557   |   0.564   |   0.578   |   0.554   |   0.669   |   0.374   |   0.556   |   0.298   |   0.398   |   0.386   |   0.397   |   0.387   |   0.388   |   0.665   |   0.809   |   0.769   |   0.768    |   0.749    |   0.843   |   0.724   |   0.447   |   0.792   |   0.400   |
|  GRASPED   |   0.745   |   0.852   |   0.883   |   0.807   |   0.910   |   0.934   |   0.921   |   0.839   |   0.700   |   0.528   |   0.534   |   0.431   | **0.462** | **0.474** |   0.458   | **0.481** | **0.480** |   0.735   |   0.784   |   0.748   |   0.711    |   0.716    |   0.848   |   0.264   |   0.624   |   0.257   |   0.454   |
|    GAMA    |   0.826   |   0.905   |   0.905   |   0.869   |   0.942   |   0.948   |   0.959   |   0.917   |   0.773   | **0.550** | **0.582** |   0.460   |   0.451   |   0.462   | **0.480** |   0.456   |   0.453   |   0.745   |   0.935   | **0.849** | **0.797**  |   0.783    |   0.921   |   0.744   |   0.727   |   0.869   |   0.523   |
|    DRL     |   0.633   |   0.620   |   0.624   |   0.676   |   0.686   |   0.707   |   0.741   |   0.683   |   0.569   |   0.376   |   0.405   |   0.296   |   0.397   |   0.397   |   0.394   |   0.396   |   0.391   |   0.416   |   0.511   |   0.398   |   0.389    |   0.394    |   0.459   |   0.882   |   0.524   |   0.936   |   0.412   |
|    WAKE    | **0.905** | **0.926** | **0.942** | **0.937** | **0.959** | **0.961** | **0.971** | **0.957** | **0.888** |   0.463   |   0.539   |   0.415   |   0.404   |   0.403   |   0.410   |   0.403   |   0.401   | **0.803** | **0.941** |   0.822   |   0.747    |   0.702    | **0.930** | **0.927** |   0.573   | **0.968** |   0.455   |

_F-score_ of attribute-level anomaly detection:

|            | Gigantic  |   Huge    |   Large   |  Medium   |    P2P    |   Paper   |   Small   |   Wide    |  BPIC12   | BPIC13\_C | BPIC13\_I | BPIC13\_O | BPIC15\_1 | BPIC15\_2 | BPIC15\_3 | BPIC15\_4 | BPIC15\_5 |  BPIC17   | BPIC20\_D | BPIC20\_I | BPIC20\_PE | BPIC20\_PR | BPIC20\_R |  Billing  |  Receipt  |   RTFMP   |  Sepsis   |
|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:----------:|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Likelihood |   0.470   |   0.571   |   0.587   |   0.559   |   0.647   |   0.645   |   0.659   |   0.569   |   0.531   |   0.102   |   0.080   |   0.129   |   0.117   |   0.108   |   0.140   |   0.137   |   0.124   |   0.174   |   0.610   |   0.505   |   0.438    |   0.514    |   0.606   | **0.640** |   0.278   | **0.691** |   0.380   |
|    DAE     |   0.193   |   0.216   |   0.216   |   0.245   |   0.257   |   0.244   |   0.261   |   0.255   |   0.291   |   0.218   |   0.147   |   0.183   |   0.102   |   0.101   |   0.084   |   0.122   |   0.158   |   0.234   |   0.361   |   0.209   |   0.200    |   0.209    |   0.330   |   0.432   |   0.170   |   0.511   |   0.184   |
|    VAE     |   0.184   |   0.206   |   0.205   |   0.233   |   0.226   |   0.200   |   0.211   |   0.246   |   0.338   |   0.193   |   0.158   |   0.191   |   0.081   |   0.108   |   0.080   |   0.107   |   0.130   |   0.135   |   0.202   |   0.154   |   0.159    |   0.159    |   0.185   |   0.416   |   0.147   |   0.563   |   0.181   |
|    LAE     |   0.344   |   0.394   |   0.430   |   0.436   |   0.529   |   0.533   |   0.510   |   0.447   |   0.477   |   0.226   |   0.207   |   0.211   |   0.184   |   0.252   |   0.197   |   0.355   |   0.234   |   0.169   |   0.497   |   0.447   |   0.457    |   0.487    |   0.479   |   0.452   |   0.385   |   0.439   |   0.320   |
|  BINetv2   |   0.308   |   0.342   |   0.329   |   0.320   |   0.346   |   0.335   |   0.362   |   0.323   |   0.504   |   0.072   |   0.258   |   0.087   |   0.020   |   0.010   |   0.049   |   0.022   |   0.013   |   0.370   |   0.571   |   0.521   |   0.502    |   0.528    |   0.561   |   0.497   |   0.141   |   0.559   |   0.060   |
|  BINetv3   |   0.322   |   0.339   |   0.340   |   0.355   |   0.336   |   0.338   |   0.365   |   0.344   |   0.470   |   0.102   |   0.264   |   0.076   |   0.021   |   0.012   |   0.022   |   0.016   |   0.019   |   0.366   |   0.535   |   0.482   |   0.468    |   0.485    |   0.553   |   0.513   |   0.157   |   0.581   |   0.066   |
|  GRASPED   |   0.579   |   0.636   |   0.646   |   0.597   |   0.582   |   0.635   |   0.593   |   0.552   |   0.503   | **0.528** | **0.406** |   0.379   |   0.385   |   0.376   |   0.410   |   0.387   |   0.398   | **0.533** |   0.468   |   0.511   |   0.498    |   0.466    |   0.507   |   0.018   |   0.479   |   0.029   |   0.324   |
|    GAMA    | **0.667** | **0.712** | **0.718** | **0.671** | **0.687** | **0.712** | **0.728** | **0.689** | **0.563** |   0.395   |   0.300   |   0.377   |   0.297   |   0.338   |   0.400   |   0.328   |   0.305   |   0.485   | **0.618** | **0.593** | **0.554**  | **0.653**  | **0.618** |   0.496   |   0.561   |   0.607   |   0.412   |
|    WAKE    |   0.600   |   0.708   |   0.710   |   0.638   |   0.644   |   0.685   |   0.667   |   0.644   |   0.491   |   0.486   |   0.261   | **0.394** | **0.402** | **0.419** | **0.455** | **0.453** | **0.437** |   0.489   |   0.372   |   0.570   |   0.466    |   0.584    |   0.467   |   0.109   | **0.564** |   0.114   | **0.474** |

Average precision (_AP_) of trace-level anomaly detection:

|            | Gigantic  |   Huge    |   Large   |  Medium   |    P2P    |   Paper   |   Small   |   Wide    |  BPIC12   | BPIC13\_C | BPIC13\_I | BPIC13\_O | BPIC15\_1 | BPIC15\_2 | BPIC15\_3 | BPIC15\_4 | BPIC15\_5 |  BPIC17   | BPIC20\_D | BPIC20\_I | BPIC20\_PE | BPIC20\_PR | BPIC20\_R |  Billing  |  Receipt  |   RTFMP   |  Sepsis   |
|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:----------:|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|   Naive    |   0.794   |   0.893   |   0.906   |   0.871   |   0.887   |   0.894   |   0.893   |   0.899   |   0.321   |   0.345   |   0.344   |   0.301   |   0.263   |   0.252   |   0.260   |   0.252   |   0.250   |   0.355   |   0.767   |   0.623   |   0.512    |   0.658    |   0.756   |   0.694   |   0.602   |   0.851   |   0.281   |
|  Sampling  |   0.756   |   0.846   |   0.875   |   0.818   |   0.850   |   0.864   |   0.859   |   0.848   |   0.276   |   0.315   |   0.298   |   0.255   |   0.258   |   0.251   |   0.253   |   0.251   |   0.250   |   0.260   |   0.610   |   0.361   |   0.315    |   0.491    |   0.622   |   0.550   |   0.522   |   0.575   |   0.253   |
|  Leverage  |   0.827   |   0.897   |   0.907   |   0.887   |   0.887   |   0.894   |   0.893   |   0.897   |   0.449   |   0.544   |   0.502   |   0.432   |   0.000   |   0.000   |   0.000   |   0.000   |   0.000   |   0.295   |   0.785   |   0.697   |   0.561    |   0.681    |   0.760   |   0.387   |   0.698   |   0.889   |   0.305   |
| Likelihood |   0.800   |   0.927   |   0.946   |   0.905   |   0.975   | **0.982** | **0.989** |   0.932   |   0.831   |   0.274   |   0.309   |   0.211   |   0.365   |   0.344   |   0.416   |   0.375   |   0.379   |   0.504   |   0.909   |   0.855   |   0.754    |   0.829    |   0.895   |   0.821   |   0.564   |   0.893   | **0.615** |
|  W2V-LOF   |   0.617   |   0.734   |   0.775   |   0.712   |   0.752   |   0.801   |   0.781   |   0.759   |   0.402   |   0.359   |   0.370   |   0.288   |   0.302   |   0.316   |   0.373   |   0.338   |   0.340   |   0.430   |   0.472   |   0.523   |   0.521    |   0.563    |   0.483   |   0.280   |   0.618   |   0.258   |   0.363   |
| VAE-OCSVM  |   0.241   |   0.229   |   0.225   |   0.239   |   0.231   |   0.222   |   0.226   |   0.233   |   0.190   |   0.217   |   0.247   |   0.169   |   0.263   |   0.265   |   0.247   |   0.246   |   0.245   |   0.242   |   0.235   |   0.240   |   0.236    |   0.244    |   0.232   |   0.148   |   0.240   |   0.150   |   0.238   |
|    GAE     |   0.459   |   0.416   |   0.617   |   0.411   |   0.615   |   0.537   |   0.494   |   0.648   |   0.386   |   0.334   |   0.321   |   0.237   |   0.259   |   0.255   |   0.271   |   0.255   |   0.258   |   0.256   |   0.452   |   0.301   |   0.296    |   0.316    |   0.434   |   0.290   |   0.284   |   0.396   |   0.278   |
|    DAE     |   0.805   |   0.852   |   0.896   |   0.900   |   0.923   |   0.940   |   0.955   |   0.931   |   0.578   |   0.344   |   0.444   |   0.339   |   0.280   |   0.272   |   0.278   |   0.273   |   0.277   |   0.325   |   0.822   |   0.517   |   0.465    |   0.493    |   0.777   |   0.604   |   0.587   |   0.845   |   0.436   |
|    VAE     |   0.570   |   0.640   |   0.798   |   0.727   |   0.705   |   0.886   |   0.864   |   0.785   |   0.646   |   0.329   |   0.310   |   0.344   |   0.283   |   0.273   |   0.300   |   0.280   |   0.280   |   0.313   |   0.678   |   0.389   |   0.414    |   0.425    |   0.624   |   0.645   |   0.580   |   0.911   |   0.448   |
|    LAE     |   0.397   |   0.439   |   0.457   |   0.548   |   0.652   |   0.676   |   0.631   |   0.528   |   0.866   |   0.317   |   0.299   |   0.373   |   0.281   |   0.300   |   0.281   |   0.310   |   0.278   |   0.264   |   0.874   |   0.707   |   0.629    |   0.639    |   0.832   |   0.789   |   0.432   |   0.790   |   0.468   |
|  BINetv2   |   0.534   |   0.598   |   0.596   |   0.583   |   0.639   |   0.620   |   0.609   |   0.581   |   0.736   |   0.181   |   0.546   |   0.169   |   0.283   |   0.267   |   0.287   |   0.291   |   0.277   |   0.690   |   0.834   |   0.835   |   0.821    |   0.815    |   0.849   |   0.645   |   0.330   |   0.695   |   0.311   |
|  BINetv3   |   0.542   |   0.579   |   0.614   |   0.622   |   0.611   |   0.616   |   0.619   |   0.598   |   0.682   |   0.269   |   0.535   |   0.206   |   0.287   |   0.270   |   0.289   |   0.271   |   0.276   |   0.689   |   0.799   |   0.827   |   0.817    |   0.778    |   0.799   |   0.688   |   0.323   |   0.724   |   0.311   |
|  GRASPED   |   0.824   |   0.924   |   0.941   |   0.885   |   0.961   |   0.977   |   0.966   |   0.907   |   0.735   |   0.556   |   0.579   |   0.427   | **0.480** | **0.459** | **0.491** | **0.503** | **0.508** |   0.796   |   0.755   |   0.800   |   0.758    |   0.727    |   0.839   |   0.160   |   0.644   |   0.156   |   0.470   |
|    GAMA    |   0.902   |   0.959   |   0.953   |   0.935   |   0.978   |   0.980   |   0.985   |   0.964   |   0.854   | **0.559** | **0.615** | **0.452** |   0.440   | **0.459** |   0.488   |   0.470   |   0.453   |   0.800   | **0.946** | **0.916** | **0.869**  | **0.835**  | **0.942** |   0.723   | **0.775** |   0.850   |   0.552   |
|    DRL     |   0.648   |   0.623   |   0.645   |   0.697   |   0.687   |   0.688   |   0.727   |   0.686   |   0.588   |   0.330   |   0.312   |   0.197   |   0.279   |   0.295   |   0.269   |   0.291   |   0.259   |   0.340   |   0.419   |   0.281   |   0.268    |   0.296    |   0.352   |   0.875   |   0.473   |   0.915   |   0.342   |
|    WAKE    | **0.943** | **0.964** | **0.973** | **0.967** | **0.982** |   0.981   |   0.987   | **0.982** | **0.901** |   0.477   |   0.577   |   0.377   |   0.333   |   0.346   |   0.375   |   0.356   |   0.360   | **0.841** |   0.941   |   0.863   |   0.769    |   0.734    |   0.941   | **0.925** |   0.569   | **0.972** |   0.444   |

Average precision (_AP_) of attribute-level anomaly detection:

|            | Gigantic  |   Huge    |   Large   |  Medium   |    P2P    |   Paper   |   Small   |   Wide    |  BPIC12   | BPIC13\_C | BPIC13\_I | BPIC13\_O | BPIC15\_1 | BPIC15\_2 | BPIC15\_3 | BPIC15\_4 | BPIC15\_5 |  BPIC17   | BPIC20\_D | BPIC20\_I | BPIC20\_PE | BPIC20\_PR | BPIC20\_R |  Billing  |  Receipt  |   RTFMP   |  Sepsis   |
|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:----------:|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Likelihood |   0.414   |   0.500   |   0.493   |   0.511   |   0.559   |   0.555   |   0.565   |   0.523   |   0.453   |   0.044   |   0.027   |   0.060   |   0.039   |   0.042   |   0.065   |   0.050   |   0.043   |   0.100   | **0.547** |   0.428   |   0.353    |   0.433    |   0.545   | **0.533** |   0.183   | **0.609** |   0.332   | 
|    DAE     |   0.113   |   0.164   |   0.149   |   0.192   |   0.199   |   0.181   |   0.198   |   0.200   |   0.198   |   0.145   |   0.080   |   0.124   |   0.030   |   0.028   |   0.025   |   0.029   |   0.057   |   0.140   |   0.314   |   0.182   |   0.147    |   0.183    |   0.289   |   0.365   |   0.109   |   0.469   |   0.118   | 
|    VAE     |   0.110   |   0.125   |   0.117   |   0.150   |   0.133   |   0.128   |   0.137   |   0.162   |   0.271   |   0.130   |   0.092   |   0.123   |   0.025   |   0.033   |   0.020   |   0.032   |   0.043   |   0.066   |   0.142   |   0.096   |   0.093    |   0.089    |   0.133   |   0.412   |   0.080   |   0.596   |   0.107   | 
|    LAE     |   0.290   |   0.352   |   0.395   |   0.396   |   0.487   |   0.512   |   0.480   |   0.404   |   0.457   |   0.137   |   0.093   |   0.138   |   0.067   |   0.115   |   0.102   |   0.200   |   0.131   |   0.058   |   0.489   |   0.451   |   0.417    |   0.482    |   0.498   |   0.314   |   0.309   |   0.292   |   0.181   | 
|  BINetv2   |   0.258   |   0.301   |   0.288   |   0.291   |   0.312   |   0.303   |   0.331   |   0.297   |   0.425   |   0.026   |   0.140   |   0.034   |   0.007   |   0.004   |   0.016   |   0.006   |   0.005   |   0.266   |   0.485   |   0.476   |   0.440    |   0.480    |   0.497   |   0.368   |   0.070   |   0.418   |   0.033   | 
|  BINetv3   |   0.271   |   0.303   |   0.304   |   0.323   |   0.302   |   0.307   |   0.330   |   0.313   |   0.386   |   0.045   |   0.150   |   0.029   |   0.008   |   0.004   |   0.009   |   0.005   |   0.007   |   0.264   |   0.442   |   0.440   |   0.409    |   0.431    |   0.456   |   0.402   |   0.073   |   0.449   |   0.030   | 
|  GRASPED   |   0.553   |   0.626   |   0.640   |   0.567   |   0.537   |   0.603   |   0.560   |   0.516   |   0.413   | **0.473** | **0.300** |   0.313   |   0.287   |   0.252   |   0.310   |   0.298   |   0.308   | **0.461** |   0.280   |   0.462   |   0.443    |   0.455    |   0.356   |   0.005   |   0.478   |   0.015   |   0.279   | 
|    GAMA    | **0.666** | **0.718** | **0.716** | **0.682** | **0.689** | **0.721** | **0.737** | **0.690** | **0.524** |   0.350   |   0.252   |   0.332   |   0.209   |   0.250   |   0.316   |   0.246   |   0.222   |   0.422   |   0.541   | **0.592** | **0.555**  | **0.639**  | **0.573** |   0.366   | **0.555** |   0.484   |   0.363   |   
|    WAKE    |   0.566   |   0.696   |   0.706   |   0.602   |   0.608   |   0.665   |   0.636   |   0.597   |   0.398   |   0.413   |   0.171   | **0.359** | **0.306** | **0.337** | **0.396** | **0.369** | **0.353** |   0.379   |   0.226   |   0.499   |   0.374    |   0.522    |   0.303   |   0.041   |   0.543   |   0.048   | **0.445** |   


## To Cite Our Paper
```
@ARTICLE{10738505,
  author={Guan, Wei and Cao, Jian and Zhao, Haiyan and Gu, Yang and Qian, Shiyou},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Survey and Benchmark of Anomaly Detection in Business Processes}, 
  year={2025},
  volume={37},
  number={1},
  pages={493-512},
  doi={10.1109/TKDE.2024.3484159}
}
```