<div align="center">
   <img src="images/OTH_and_ReMIC_merged.png" style="background-color:white; width:100%">
</div>



# DE: Graduierung von Fazialisparesen durch Methoden des Maschinellen Lernens v0.0.0

##### EN: Grading of Facial Palsy by Machine Learning methods

## Thesis TODO and INFOS
- Until final Date
   1. generate Poster
- FINAL DATE (2x Thesis with CD)
   1. PDF of Thesis
   2. Teaser Picture
   3. Copy of Pages where i citate from (document/docs or document/ePaper)
   4. images (document/thesis/images)
   5. Source Code (copy of gitlab repro)
- Use Vector graphics
- UML Diagrams (sequence diagram, ...)

#### TODO-LIST

###### Training
- [ ] OptimizerClass/SchedulerClass to functions
   1. [ ] Adding Scheduler to list
   2. [ ] Make it available via argparse
- [ ] Dataloader
   1. [ ] Adding Dataloader Class
   2. [ ] Gather Data from the images
   3. [x] Structure of the images and labels (use template see in settings.py)
   4. [x] LoadImages Class for only loading Images
   5. [ ] Augmentation for Images (neccessary: flipping the images)
- [ ] Training function for epochs
- [ ] Process Images twice (img and inv_img) for better training results
- [ ] Decision Tree for the Grading
- [ ] Validation of training (see Plotting)
- [ ] DataParralell and DistributedDataParrallel for faster traning

###### Detection
- [ ] Input for the System (Single Patient, list of Patients, list of Patients by Category)
- [ ] ? Process Images twice (img and inv_img)
- [ ] Prediction interpreter
- [ ] Decision Tree for the Grading
- [ ] output result list

###### Other
- [ ] Optimizing Code via Numba and Lru-Cache especially frequently used functions
- [ ] Plotting Results from training/validation
   1. [ ] for each Category
   2. [ ] loss, gain, performance average, precison, confusion matrix
- [ ] Linting and Unittests
   1. [ ] Linting Code and write all #pylint: disable to a list
   2. [ ] Write Unittest code for all relevant Functions
- [ ] Adding & Checking Docstring's in all files and functions
   1. [ ] Header Docstring addes (Author, file name, License)
   2. [ ] Function Docstring
- [ ] [Python Package generation](https://packaging.python.org/tutorials/packaging-projects/)
- [ ] API Access via docker-compose/Dockerfile and FastAPI
- [ ] Add License (GNU General Public License v3.0)
- [ ] Jupyter lab for vizualizing training and testing as tutorial

## Project

#### Install
clone Repository and install Dependencies  
or  
let the Project update all Dependencies automatic if already a version is installed
~~~bash
git clone ???
cd ???
pip install -r source/requirements.txt
~~~


#### Use Cases
 --> Will be added soon! (How to start and use the Code)

###### Coding of the Pictures
1. Ruhender Gesichtsausdruck
2. Augenbrauen heben
3. Lächeln, geschlossener Mund
4. Lächeln, geöffneter Mund
5. Lippen schürzen, „Duckface"
6. Augenschluss, leicht
7. Augenschluss, forciert
8. Nase rümpfen
9. Depression Unterlippe

#### Debug
<details open>
<summary>Linting of all python files for a unified structure look using pylint Package</summary>

~~~shell
pylint source --extension-pkg-whitelist=torch --generated-members=numpy,torch --max-line-length=170
~~~
<!---
adding Reason
-->


| File                     | Function                        |   pylint disable                                   |   Reason    |
| :---                     | :----                           | :----                                              | :----       |
| detect.py                | detect                          | too-many-arguments <br> too-many-locals            |             |
| detect.py                | main                            | pointless-string-statement <br> unnecessary-lambda |             |
| train.py                 | train                           | too-many-arguments <br> too-many-locals            |             |
| train.py                 | main                            | pointless-string-statement <br> unnecessary-lambda |             |
| config.py                | n.a                             | n.a                                                |             |
| unit_test.py             | test_check_python_no_exception  | no-self-use                                        |             |
| unit_test.py             | main                            | pointless-string-statement                         |             |
|                          |                                 |                                                    |             |
| utils/dataloader.py      | Import                          | import-error                                       |             |
| utils/general.py         | n.a                             | n.a                                                |             |
| utils/pytorch_utils.py   | n.a                             | n.a                                                |             |
| utils/templates.py       | n.a                             | n.a                                                |             |

</details>

<details open>
<summary>Unittest of relevant Functions using builtin Python unittest libary and pytest Package</summary>

~~~shell
pytest source --color=no

or

cd source &&  python unit_test.py
~~~
</details>

## Other
#### Tools used for the Thesis & Project
- [atom.io v1.58.0](https://atom.io/)
   1. Included Git-Version-Management
   1. Package [atom-latex v0.9.1](https://atom.io/packages/atom-latex) for building the Thesis
   2. Package [language-latex v1.2.0](https://atom.io/packages/language-latex) for syntax highliting
   3. Package [platformio-ide-terminal v2.10.1](https://atom.io/packages/platformio-ide-terminal) Terminal access
- [Python v3.8.6](https://www.python.org/)
- Python Packages (look at source/requirements.txt for more informations about)
