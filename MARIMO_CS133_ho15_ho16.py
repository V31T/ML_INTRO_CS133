import marimo

__generated_with = "0.4.3"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        <a href="https://colab.research.google.com/github/V31T/ML_INTRO_CS133/blob/main/HENRY_PHAM_CS133_ho15_ho16.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## **Hands-on 15**

        ### **Goal**:
        You are the data scientist assigned to perform the data pre-processing and preparing the data for Machine Learning algorithms.

        1. perform data exploration to understand the data (2.5 points)
        2. prepare the test and training sets. (2.5 points)
        3. pre-processing of the data, including fixing all the missing values (set the missing values to median values) and any other ones that you think are appropriate to perform. Build a pipeline to perform data transformation. (5 points)

        In the next hands-on, we will use 14 out of 15 attributes as pedictors describe below to predict if income goes above or below \$50K/yr based on census data. `Income` will be the label.

        ### Data:
        An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education level, age, gender, occupation, and etc.

        ### Fields:
        The dataset contains 15 columns

        #### Target field: Income
        - The income is divide into two classes: 50K

        #### Number of attributes: 14
        -- These are the demographics and other features to describe a person

        We can explore the possibility in predicting income level based on the individual’s personal information

        - `age`: continuous.
        - `workclass`: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
        - `fnlwgt`: continuous.
        - `education`: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
        - `education-num`: continuous.
        - `marital-status`: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
        - `occupation`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
        - `relationship`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
        - `race`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
        - `sex`: Female, Male.
        - `capital-gain`: continuous.
        - `capital-loss`: continuous.
        - `hours-per-week`: continuous.
        - `native-country`: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
        - `salary`: >50K,<=50K

        Note: "?" is used to represent missing data in this dataset.
        """
    )
    return


app._unparsable_cell(
    r"""
    import pandas as pd
    %matplotlib inline
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Read in data
    adults = 'https://raw.githubusercontent.com/csbfx/advpy122-data/master/adult.csv'
    df = pd.read_csv(adults, na_values=['?'])

    df
    """,
    name="__"
)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Hands-on 15 Part 1: Data Exploration (2.5 points)
        """
    )
    return


@app.cell
def __(df):
    # Your code here . . .
    df.info()
    return


@app.cell
def __(df):
    df["income"].value_counts()
    return


@app.cell
def __(df):
    df["gender"].value_counts()
    return


@app.cell
def __(df):
    df["workclass"].value_counts()
    return


@app.cell
def __(df):
    df["education"].value_counts()
    return


@app.cell
def __(df):
    df["occupation"].value_counts()
    return


@app.cell
def __(df):
    df["marital-status"].value_counts()
    return


@app.cell
def __(df, plt):
    ##from matplot libraries
    df.hist(bins=50, figsize=(15,10))
    plt.show()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        **Immediate Observations**  

        more than 75% of dataset has >50k income, could assume that any columns that have a single high value count share a strong correlation to the person's income. So far, just off counting values, Gender, workforce, hours-per-week and marital status seem to have a high impact on income level. Existence of capital gain or loss could have a strong impact on income level too.  

        occupation, workclass, and native-country have null values

        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Hands-on 15 Part 2: Prepare Training & Testing data sets (2.5 points)
        """
    )
    return


@app.cell
def __():
    # Your code here . . .


    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Hands-on 15 Part 3: Pre-processing data (5 points)
        """
    )
    return


@app.cell
def __():
    # Your code here . . .
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## **Hands-on 16**
        Use the results from Hands-on 15 for the following steps:

        1. Select ML Models and perform 10-fold Cross Validation. (5 points)
        2. Pick the best model from step 1 and perform fine-tuning. (2.5 points)
        3. Test ML model with the test set. (2.5 points)
        4. Bonus: Create a plot with ROC curves to compare the performance of the ML models that you have trained using different ML classifiers. (2 points)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Hands-on 16 Part 1: Select ML Models, perform 10-fold Cross Validation (5 points)
        Try four different ML models for classification.
        """
    )
    return


@app.cell
def __():
    # Your code here . . .
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Hands-on 16 Part 2: Pick the best model from Part 1 and perform fine-tuning (2.5 points)
        """
    )
    return


@app.cell
def __():
    # Your code here . . .
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Test ML model with the test set (2.5 points)
        Use the fine-tuned model and evaluate its performance using the test set that you have created in Hands-on 15.
        """
    )
    return


@app.cell
def __():
    # Your code here . . .
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Bonus: Plot the ROC curve to compare the performace of the ML classifiers (1  point)
        """
    )
    return


@app.cell
def __():
    # Your code here . . .
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Bonus: Evaluating the model using the Confusion Matrix and a Precision-Recall Curve (1 point)
        A confusion matrix is a tabular summary of the number of correct and incorrect predictions made by a classifier. It can be used to evaluate the performance of a classification model through the calculation of performance metrics such as [accuracy, precision, recall, and F1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html). Here is [an article](https://medium.com/swlh/explaining-accuracy-precision-recall-and-f1-score-f29d370caaa8) that gives a good explaination of Precision, Recall, and F1-score.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Accuracy

        Accuracy = $\frac{True\ Positives\ +\ True\ Negatives}{All\ Samples}$

        ### Precision (aka Specificity)

        Precision = $\frac{True\ Positives}{True\ Positives\ +\ False\ Positives}$
        = $\frac{True\ Positives}{Total\ Predicted\ Positives}$


        ### Recall (aka Sensitivity)

        Recall = $\frac{True\ Positives}{True\ Positives\ +\ False\ Negatives}$
        = $\frac{True\ Positives}{Total\ Actual\ Positives}$

        ### F1-score (combining Precision and Recall)

        F1-score = $\frac{2\ ×\ (Precision\ ×\ Recall)}{Precision\ +\ Recall}$
        """
    )
    return


@app.cell
def __():
    # Your code here . . .
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Precision-Recall Curve
        [Precision-Recall Curve documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py)

        """
    )
    return


@app.cell
def __():
    # Your code here . . .

    return


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()

