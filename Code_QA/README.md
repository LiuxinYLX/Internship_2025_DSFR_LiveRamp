# Architecture of the projects

```
My_project/ <br/>
├── main.py <br/>                 
├── config/                  # Configuration, parameters
│   └── __init__.py       
│   └── configuration.py
│   └── table.py       
├── modules/                 # Functions
│   ├── __init__.py         
│   ├── generateQuery.py        
│   ├── standarize.py
│   ├── generateDictionary.py             
├── data/                    # Data（.csv/.json）
└── README.md                    
```

# Explanation of pipeline
## Step1: Data cleaning
### 1.1 Format-level Cleaning
1. Convert multiple space to single space ```REGEXP_REPLACE(_, r'\\s+', ' ')```
2. Convert to upper case ```UPPER()```
3. Delete spaces before and after ```TRIM()```
4. Replace diacritics (accents) ```REGEXP_REPLACE(NORMALIZE(_, NFD), r'\pM', '')```
### 1.2 Semantic-level Cleaning
5. Handle missing values (convert variants like "N/A", "null", "" to NULL)
6. Unify the brands' names:

> To begin with, we remove all non-alphanumeric characters from brand names to normalize similar variants (e.g., "L'OREAL", "L OREAL" → "LOREAL").  
>
> Next, we group the original brand names based on their normalized form and count the frequency of each original name within each group.  
>
> We then select the most frequent original name in each group as the standardized brand name and store this mapping in a dictionary.
> 
> As a result, all variants like "L'OREAL" and "L OREAL" will be unified under the most common form depending on the frequency.  



## Step2: Data validaton
1. Check duplicates across the entire line
2. Check duplicates based on primary key
3. Check barcode length
4. [Option] Check date