#TODO Docstring
"""
TODO
"""

import os
import io
import sqlite3
from functools import lru_cache
import torch

from .singleton import Singleton #pylint: disable=import-error
from .decorators import try_except_none, try_except, thread_safe #pylint: disable=import-error
from .config import LRU_MAX_SIZE, LOGGER

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

@lru_cache(LRU_MAX_SIZE)
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
        :param conn: connection cursor
        """
        self.prefix_for_log = ""
        self.db_file = 'cache.db'
        self.conn = None

    @try_except
    def delete(self):
        """
        Destructor: remove database
        """
        if self.conn is not None:
            self.conn.close()
            LOGGER.info("%sClosed Connection to Database", self.prefix_for_log)
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
            LOGGER.info("%sDeleted Database File (Cache)!", self.prefix_for_log)

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

    @try_except_none
    def create_db_connection(self):
        """
        create a database connection to the SQLite database specified by db_file
        :return: Connection object or None
        """
        self.conn = sqlite3.connect(self.db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        return self.conn

    @try_except
    def create_db_table(self, table):
        """
        create a table from the table statement
        :param table: a CREATE TABLE statement
        """
        cursor = self.conn.cursor()
        cursor.execute(table)

    @try_except_none
    def db_table_entries_exists(self, table):
        """
        Checks if Table exixst
        :param table: Table
        :return True or False (bool)
        """
        cursor = self.conn.cursor()
        item = cursor.execute("SELECT * FROM "+table).fetchall()
        return bool(item)

    @thread_safe
    @try_except
    def insert_db(self, table, params, param_question):
        """
        inserting item in the table and commit it to database
        :param table: Inserting item in Table
        """
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO "+table+" VALUES " + param_question ,params)
        self.conn.commit()

    @lru_cache(LRU_MAX_SIZE)
    @try_except_none
    def get_db_one(self, table, idx):
        """
        retriving one item from the table
        :param table: Retriving from Table
        :param idx: index (int)
        """
        cursor = self.conn.cursor()
        item = cursor.execute("SELECT * FROM "+table+" WHERE id=?",(idx,)).fetchone()
        return item

    @lru_cache(LRU_MAX_SIZE)
    @try_except_none
    def get_db_all(self, table):
        """
        retriving all items from the table
        :param table: Retriving from Table
        """
        cursor = self.conn.cursor()
        item = cursor.execute("SELECT * FROM "+table).fetchall()
        return item
