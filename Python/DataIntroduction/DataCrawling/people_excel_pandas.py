import pandas as pd

fileName = "people_data.xlsx"
sheet_name = "Sheet0"

book = pd.read_excel(fileName, sheetname = sheet_name, header=3)
book = book.sort_values(by=2016, ascending=False)

print(book)