 ![Language](https://img.shields.io/badge/language-python--3.7-blue) [![Contributors][contributors-shield]][contributors-url] [![Forks][forks-shield]][forks-url] [![Stargazers][stars-shield]][stars-url] [![Issues][issues-shield]][issues-url] [![MIT License][license-shield]][license-url] [![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier">
    <img src="results/logo.jpg" alt="Logo" width="80" height="80">
  </a>
  <h3 align="center">Pattern Recognition and Neural Networks</h3>
  <p align="center">
    Classifiers and Sentiment analysis
    <br />
    <a href=https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier><strong>Explore the repository»</strong></a>
    <br />
    <br />
    <a href=https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier/blob/master/Problem_Statement.pdf>View Problem Statement</a>
    <a href=https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier/blob/master/results/report.pdf>View Report</a>
  </p>




</p>

> tags : bayes classifier, nearest neighbour classifier, EM algorithm, gaussian mixtures, density estimation, sentiment analysis, maximum likelihood estimation, tfidf, bag of words 



<!-- ABOUT THE PROJECT -->

## About The Repository

This repository holds the python implementation files for Assignment #1 for E1 213 Pattern Recognition and Neural Networks offered at the Indian Institute of Science (IISc), Bangalore. In this assignment we will explore and compare different methods of classifiers. In all the problems, we have some data on which  2-class classifiers have to be learnt, and then its performance has to be assessed using a test set. The following methods have been implemented across the problems.

* Bayes classifier
* Nearest neighbour classifier
* EM algorithm for Gaussian mixtures
* Density estimation
* Maximum Likelihood Estimation 
* Sentiment analysis (Movie review)

Problem 1 deals with the implementation of 2-class Bayes classifier, and nearest neighbour classifier using the values estimated by maximum likelihood and EM algorithm for GMM. Problem 2 deals with the implementation of 2-class Bayes classifier using the values estimated by maximum likelihood and nearest neighbour classifier.  Problem 3 deals with implementation of Bayes classifier for Gaussian mixture class conditional densities. Problem 4 deals with the implementation of sentiment analysis of movie reviews using bag of words approach and TF-IDF approach. 



### Built With
This project was built with 

* python v3.7
* The list of libraries used for developing this project is available at [requirements.txt](requirements.txt).



<!-- GETTING STARTED -->

## Getting Started

Clone the repository into a local machine using

```shell
git clone https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier
```

### Prerequisites

Please install required libraries by running the following command (preferably within a virtual environment).

```shell
pip install -r requirements.txt
```

### Instructions to run

There are four python files - `problem_1.py`, `problem_2.py`, `problem_3.py` , and `problem_4.py` - each corresponding to the particular problem in the Problem Statement. Each problem has their corresponding implementation files under a python package with the same name. Each package has python modules and functions to load data, train a model, test it, and write the performance metrics to an output file at `./results` with the same file name. 

##### Running the program

```shell
python problem_<QUES_NUM>.py
```



<!-- RESULTS -->

## Results

View [Report](results/report.pdf) for the results and detailed discussions.



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Vineeth S  - vs96codes@gmail.com

Project Link: [https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier](https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier.svg?style=flat-square
[contributors-url]: https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier.svg?style=flat-square
[forks-url]: https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier/network/members
[stars-shield]: https://img.shields.io/github/stars/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier.svg?style=flat-square
[stars-url]: https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier/stargazers
[issues-shield]: https://img.shields.io/github/issues/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier.svg?style=flat-square
[issues-url]: https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier/issues
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/vineeths96/Bayes-classifier-and-Nearest-Neighbour-classifier/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/vineeths

