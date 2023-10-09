#a call with paranthesis like head() means it is a method of that class, howeve a call without paranthesis like shape is an attribute of that class
import pandas as pd

dict_1={"car" : ["city", "ignis", "800", "Verna", "Venue", "Punto"], "brand": ["Honda", "maruti", "Maruti", "Hyundia","Hyundai", "Fiat"],"cost":[990000,600000,100000,800000,950000,750000],"year": [5,7,10,4,2,6]}

#turn dictionary to dataframe
auto=pd.DataFrame(dict_1)
print(auto)

#read a csv file containing data
data=pd.read_csv("titanic.csv")

#head function displays top five rows of the dataset you can pass as an argument number of rows you want to see
print(data.head())

#info function gives you information about columns, data inside them, their data types, and null values
data.info()

#shape function will give you number of rows and colums
print(data.shape)

#columns will show the labels of all the columns
print(data.columns)

#when you use value_counts for a specific column it gives you the number of each value introduced to this column and its datatype
print(data['Embarked'].value_counts())

#unique will just return the unique values introduced to this column
print(data["Sex"].unique())


#Accessing Columns
print(data['Name'])
print(data[['Name', 'Sex']])


#Accessing Rows
#display complete first row with all columns
print(data.iloc[0,:].values)
#display first 5 rows with all columns
print(data.iloc[0:5,:].values)
#display first 5 rows with one column
print(data.iloc[0:5,4].values)

#loc method is to use the name of the columns
print(data.loc [1:5, ['Name', "Sex"]])



#Filtering of data
new_data=data["Sex"]=="male"
print(new_data)
isna = auto['car'].isna()
print(auto.loc[isna])
notna = auto['car'].notna()
print(notna)



#Sorting of Data
#sorted in descending order
print(auto.sort_values("cost"))



#Grouping And Aggregation
#grouped the data according to the sex so females and males
groups = data.groupby(['Sex'])
#you get the mean for males alone and for females alone
print(groups.mean(numeric_only=True))
print(groups.size())
print(groups.count())