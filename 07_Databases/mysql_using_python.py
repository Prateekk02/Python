import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "abc",
    password = "password"
)
mycursor = mydb.cursor()
mycursor.execute("create database if not exists data")
mycursor.execute("create table if not exists data.table_data(c1 int, c2 varchar(50), c3 int, c4 varchar(30));")
mycursor.execute("insert into data.table_data values(1, 'Anakin', 66, 'Skywalker');")
mydb.commit()

# For getting all the column
"""
mycursor.execute("select * from data.table_data")
for i in mycursor.fetchall():
    print(i)
"""
# For getting specific column.
mycursor.execute("select c1, c4 from data.table_data")
for i in mycursor:
    print(i)
mycursor.close()