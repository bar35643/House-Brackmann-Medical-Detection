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
   2. Poster
   3. Copy of Pages where i citate from (document/docs or document/ePaper)
   4. images (document/thesis/images)
   5. Source Code (copy of gitlab repro)
- Use Vector graphics
- UML Diagrams (sequence diagram, ...)

#### TODO-LIST
- [ ] Optimizing Code via Numba and Lru-Cache especially frequently used functions
- [ ] OptimizerClass/SchedulerClass to functions
- [ ] Dataloader
   1. [ ] Adding Dataloader Class
   2. [ ] Gather Data from the images
   3. [ ] Structure of the images and labels (use template see in settings.py)
   4. [ ] LoadImages Class for only loading Images
- [ ] Decission Tree Training for the Grading
- [ ] Plotting Results from training
   1. [ ] for each Category
   2. [ ] loss, gain, performance average, precison, confusion matrix




## Project

#### Install
clone Repository and install Dependencies  
or  
let the Project gather all Dependencies automatic
~~~bash
git clone ???
cd ???
pip install -r source/requirements.txt
~~~


#### Use Cases
 --> Will be added soon! (How to start and use the Code)

<!---
#### Changelog
- Template 1
   1. ??
   2. ??
- Template 2
   1. ??
   2. ??

-->

#### Debug
<details open>
<summary>Linting of all python files for a unified structure</summary>

~~~shell
pylint source --extension-pkg-whitelist=torch --max-line-length=150
~~~
<!---
Modifications in Files:
- abc
- abc
-->
</details>

<details open>
<summary>Unittest usind builtin Python libary</summary>

~~~shell
python unit_test.py
~~~
</details>

## Other
#### Tools used for the Thesis & Project
- [atom.io v1.58.0](https://atom.io/)
   1. Included Git-Version-Management
   1. Package [atom-latex v0.9.1](https://atom.io/packages/atom-latex) for building the Thesis
   2. Package [language-latex v1.2.0](https://atom.io/packages/language-latex) for syntax highliting
- [Python v3.8.6](https://www.python.org/)
- Python Packages (look at source/requirements.txt for more informations about)
