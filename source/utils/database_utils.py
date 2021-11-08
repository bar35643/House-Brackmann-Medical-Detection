#TODO Docstring
"""
TODO
"""

import io
import sqlite3
from sqlite3 import Error
import torch
from .singleton import Singleton #pylint: disable=import-error

def adapt_dictionary(data):
    """
    Database functions to convert np.array to entry
    :param data: dict
    :return: binary stream
    """
    out = io.BytesIO()
    torch.save(data, out)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_dictionary(text):
    """
    Database functions to convert entries back to np.np_array
    :param text: text
    :return: dict
    """
    out = io.BytesIO(text)
    out.seek(0)
    return torch.load(out)

sqlite3.register_adapter(dict, adapt_dictionary)
sqlite3.register_converter("dict", convert_dictionary)

@Singleton
class Database():
    """
    Database Class for wrapping the sqlite3
    """
    def __init__(self):
        """
        Initializes Database Class

        :param db_file: File name (str)
        :param prefix_for_log: logger output prefix (str)
        """
        self.prefix_for_log = ""
        self.db_file = 'pythonsqlite.db'
        self.conn = None

    def set(self, db_file:str, prefix_for_log:str):
        """
        set Class items
        :param db_file: database file name (str)
        :param prefix_for_log: logger output prefix (str)
        """
        self.prefix_for_log = prefix_for_log
        self.db_file=db_file

    def get_conn(self):
        """
        return the connection
        :return: Connection object or None
        """
        return self.conn

    def create_db_connection(self):
        """
        create a database connection to the SQLite database specified by db_file
        :return: Connection object or None
        """
        try:
            self.conn = sqlite3.connect(self.db_file, detect_types=sqlite3.PARSE_DECLTYPES)
            return self.conn
        except Error as err:
            print(err)

        return self.conn


    def create_db_table(self, table):
        """
        create a table from the table statement
        :param table: a CREATE TABLE statement
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(table)
        except Error as err:
            print(err)

    def insert_db(self, table, params, param_question):
        """
        inserting item in the table and commit it to database
        :param table: Inserting item in Table
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO "+table+" VALUES " + param_question ,params)
            self.conn.commit()
        except Error as err:
            print(err)

    def db_table_entries_exists(self, table):
        """
        Checks if Table exixst
        :param table: Table
        :return True or False (bool)
        """
        try:
            cursor = self.conn.cursor()
            item = cursor.execute("SELECT * FROM "+table).fetchall()
        except Error as err:
            print(err)

        if item:
            return True

        return False

    def get_db_one(self, table, idx):
        """
        retriving one item from the table
        :param table: Retriving from Table
        :param idx: index (int)
        """
        try:
            cursor = self.conn.cursor()
            item = cursor.execute("SELECT * FROM "+table+" WHERE id=?",(idx,)).fetchone()
            return item[1], item[2]
        except Error as err:
            print(err)

        return None

    def get_db_all(self, table):
        """
        retriving all items from the table
        :param table: Retriving from Table
        """
        try:
            cursor = self.conn.cursor()
            item = cursor.execute("SELECT * FROM "+table).fetchall()
            return item
        except Error as err:
            print(err)

        return None
