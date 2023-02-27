# Machine-Learning-Model-with-Python-Scikit-learn
This machine learning model uses a retail website's customer data to determine who would be interested in a new promotion.
3 datasets were used to train the model: 
  - a users .csv that contained user information such as user id, age, and previous purchase amount
  - a logs .csv that contained information about how many webpages were visited by the user and how long they were on each page
- a .csv containing the user id and a y column. The y column value was either a 1, meaning they clicked the email, or 0, meaning they didn't click the email

The UserPredictor class fits and predicts whether a user will click on the new promotion when given the training and testing datasets. The predict function will return a numpy array with True and False values. If the value is True at the index, that means the model correctly predicted whether or not the user clicked on the promotion.
I unfortunately built this project on a virtual machine that I no longer have access to and can't access the datasets I used for this model or the .ipynb I used to test the methods.
