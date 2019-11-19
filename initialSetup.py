from phase1.textFileProcessor import TextFileProcessor
from phase1.xmlFileProcessor import XmlFileProcessor
from dbUtils import DBUtils

# Sets up the sql file if it does not exist. Processes text files and inserts data into sql tables
class InitialSetup:

    def create_initial_tables(self):
        queryList = ["CREATE TABLE users (userId text PRIMARY KEY)",
                     "CREATE TABLE images (imageId text PRIMARY KEY)",
                     "CREATE TABLE locations (locationId integer  PRIMARY KEY, locationKey text, latitude REAL, "
                     "longitude REAL, wiki text, locationName text)",
                     "CREATE TABLE terms (term text PRIMARY KEY)",
                     "CREATE TABLE termsPerUser (term text NOT NULL, "
                     "userId text NOT NULL, TF integer, DF integer, TF_IDF REAL, PRIMARY KEY(term, userId))",
                     "CREATE TABLE termsPerImage (term text NOT NULL, "
                     "imageId text NOT NULL, TF integer, DF integer, TF_IDF REAL, PRIMARY KEY(term, imageId))",
                     "CREATE TABLE termsPerLocation (term text NOT NULL, "
                     "locationId integer NOT NULL, TF integer, DF integer, TF_IDF REAL, PRIMARY KEY(term, locationId))"
                     ]
        self._database_operations.executeWriteQueries(queryList)

    def __init__(self, database_operations, base_path):
        self._base_path = base_path
        self._textTermsPerUserFile = "/desctxt/devset_textTermsPerUser.txt"
        self._textTermsPerImageFile = "/desctxt/devset_textTermsPerImage.txt"
        self._textTermsPerPOIwFolderNamesFile = "/desctxt/devset_textTermsPerPOI.wFolderNames.txt"
        self._devset_topics = "/devset_topics.xml"
        self._database_operations = database_operations
        self.uniqueTerms = set();


    def process_text_terms_per_user(self, line):
        # regex = "^[\w]+@[\w]+\s\".+\"\s\d+\s\d+\s\d+\.\d+$"
        # compiledPattern = re.compile(regex);
        quoteIndex = line.find("\"");
        userId = line[:quoteIndex]
        userId = userId.strip();
        insertUserQuery =    "INSERT into users(userId) values(\"%s\")" %userId
        term_data = line[quoteIndex:]
        insertTermPerUserQuery = "INSERT into termsPerUser(term, userId, TF, DF, TF_IDF) values(%s, \"%s\", %d, %d, %f)"
        queries = self.process_text_file_generic(term_data, userId, insertTermPerUserQuery)
        queries.append(insertUserQuery)
        return queries
        # self._database_operations.executeWriteQueries(queries)

    def process_text_terms_per_image(self, line):
        # regex = "^[\w]+@[\w]+\s\".+\"\s\d+\s\d+\s\d+\.\d+$"
        # compiledPattern = re.compile(regex);
        quoteIndex = line.find("\"");
        imageId = line[:quoteIndex]
        imageId = imageId.strip();
        insertImageQuery = "INSERT into images(imageId) values(\"%s\")" % imageId
        term_data = line[quoteIndex:]
        insertTermPerImageQuery = "INSERT into termsPerImage(term, imageId, TF, DF, TF_IDF) values(%s, \"%s\", %d, %d, %f)"
        queries = self.process_text_file_generic(term_data, imageId, insertTermPerImageQuery)
        queries.append(insertImageQuery)
        return queries
        # self._database_operations.executeWriteQueries(queries)

    def process_text_terms_per_POI(self, line, helper_object):
        # regex = "^[\w]+@[\w]+\s\".+\"\s\d+\s\d+\s\d+\.\d+$"
        # compiledPattern = re.compile(regex);
        quoteIndex = line.find("\"");
        location_data = line[:quoteIndex]
        location_data_split = location_data.split(" ", 1)
        location_key = location_data_split[0]
        location_name = location_data_split[1]
        location_name = location_name.strip();
        location_id = helper_object[location_key]
        insertLocationQuery = "UPDATE locations set locationName = \"{0}\" where locationKey=\"{1}\"".format(location_name, location_key)
        term_data = line[quoteIndex:]
        insertTermPerLocationQuery = "INSERT into termsPerLocation(term, locationId, TF, DF, TF_IDF) values(%s, \"%s\", %d, %d, %f)"
        queries = self.process_text_file_generic(term_data, location_id, insertTermPerLocationQuery)
        queries.append(insertLocationQuery)
        return queries
        # self._database_operations.executeWriteQueries(queries)

    def process_text_file_generic(self, term_data, primary_metric, insert_term_per_metric_query):
        queries = []
        while term_data:
            text_indicators = term_data.split(" ", 4)
            if len(text_indicators) == 5:

                term = text_indicators[0]
                TF = int(text_indicators[1])
                DF = int(text_indicators[2])
                TF_IDF = float(text_indicators[3])
                term_data = text_indicators[4];

                if term not in self.uniqueTerms:
                    self.uniqueTerms.add(term)
                    insertTermQuery = "INSERT into terms(term) values(%s)" % term
                    queries.append(insertTermQuery)

                insertTermPerMetricQuery = insert_term_per_metric_query % (
                term, primary_metric, TF, DF, TF_IDF)

                queries.append(insertTermPerMetricQuery)
            else:
                term_data = None;
        return  queries;

    def process_devset_topics_xml(self, root):
        queries = []
        for l1child in root:
            loc_obj = {}
            for l2child in l1child:
                loc_obj[l2child.tag] = l2child.text

            insertLocationQuery = "INSERT into locations(locationId, locationKey, latitude, longitude, wiki) values({0}," \
                                  " \"{1}\", {2}, {3},\"{4}\")".format(int(loc_obj['number']), loc_obj['title'],
                                                                       float(loc_obj['latitude']),
                                                                             float(loc_obj['longitude']),
                                                                             loc_obj['wiki'], "")
            queries.append(insertLocationQuery)
        self._database_operations.executeWriteQueries(queries)


    def process_desctxt_files(self):

        text_processor = TextFileProcessor(self._base_path)
        xml_processor = XmlFileProcessor(self._base_path)

        xml_processor.parse_xml(self._devset_topics, self.process_devset_topics_xml)
        queries = text_processor.process_text_file(self._textTermsPerUserFile, self.process_text_terms_per_user)
        queries += text_processor.process_text_file(self._textTermsPerImageFile, self.process_text_terms_per_image)
        queries += text_processor.process_text_file(self._textTermsPerPOIwFolderNamesFile, self.process_text_terms_per_POI, DBUtils.create_location_key_id_map(self._database_operations))
        self._database_operations.executeWriteQueries(queries)

    def setup_database_from_devset_data(self):
        self.create_initial_tables();
        self.process_desctxt_files()
        return None;