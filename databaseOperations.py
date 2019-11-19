
import _sqlite3 as sqlite;

# Read and write to sqlite3 database
class DatabaseOperations:
    def __init__(self, sql_file):
        self.__filePath = sql_file

    # Execute Write Queries
    def executeWriteQueries(self, queries):
        try:
            conn = sqlite.connect(self.__filePath);
            c = conn.cursor();
            for query in queries:
                try:
                    c.execute(query)
                except sqlite.Error as e:
                    pass
            conn.commit();
        except sqlite.Error as e:
            print(e)
        finally:
            conn.close()

    def executeSelectQueryWithRowProcessing(self, query, row_processor, row_processing_params, executePragmaForPerformance = False):
        try:
            conn = sqlite.connect(self.__filePath);
            c = conn.cursor();
            if executePragmaForPerformance:
                'Used to set sqllite memory allocation from swap space to main memory to avoid memory overflow'
                c.execute("pragma temp_store = 2")
            c.execute(query)
            self.queryResultGenerator(c, row_processing_params, row_processor)
        except sqlite.Error as e:
            print(e)
        finally:
            conn.close()

    def queryResultGenerator(self, cursor, row_processing_params, row_processor, arraysize=1000000):
        'An iterator that uses fetchmany to keep memory usage down'
        while True:
            results = cursor.fetchmany(arraysize)
            if not results:
                break
            for item in results:
                row_processor(row_processing_params, item)


    # Execute Select QUery for the database
    def executeSelectQuery(self, query, executePragmaForPerformance = False):
        try:
            conn = sqlite.connect(self.__filePath);
            c = conn.cursor();
            if executePragmaForPerformance:
                c.execute("pragma temp_store = 2")
            c.execute(query)
            return c.fetchall()
        except sqlite.Error as e:
            print(e)
        finally:
            conn.close()
