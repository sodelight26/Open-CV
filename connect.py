import mysql.connector

mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="opencv"
)
if mydb:
    print("successfully!")
else:
    print("fail!")