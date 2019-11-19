import math;

from dbUtils import DBUtils
from objects.dotProduct import DotProduct
from objects.termInfo import TermInfo


# Processes user queries
class QueryProcessor:

    def __init__(self, database_operations):
        self._database_operations = database_operations

    # Generic function to process results
    def process_text_result_sets(self, result, k, source_word_dict, conversion_func):
        common_words_dict = {}
        uncommon_words_dict = {}

        for row in result:
            arg = row[0]
            if row[3]:
                if arg not in common_words_dict:
                    common_words_dict[arg] = []
                term = row[1]
                common_words_dict[arg].append(TermInfo(term, conversion_func(row[2])))
            else:
                if arg not in uncommon_words_dict:
                    uncommon_words_dict[arg] = []
                uncommon_words_dict[arg].append(conversion_func(row[2]))

        dot_product_list = self.calculateKHighestScores(source_word_dict, common_words_dict, uncommon_words_dict, k)
        return dot_product_list;

    def display_result(self, dot_product_ranked_list):
        dot_product_ranked_list = sorted(dot_product_ranked_list, key=lambda dot_prod_value: dot_prod_value.value,
                                         reverse=True)
        for dot_product in dot_product_ranked_list:
            print(dot_product.arg + " " + str(dot_product.value))
            print("Highest contributing terms are ")
            for contributing_term in dot_product.contributing_terms: print(contributing_term.term)

    #  Displays the result by printing loc name using key as input
    def display_location_result(self, dot_product_ranked_list, location_map):
        for dot_product in dot_product_ranked_list:
            location = location_map[dot_product.arg]
            print(location + " " + str(dot_product.value))
            print("Highest contributing terms are ")
            for contributing_term in dot_product.contributing_terms: print(contributing_term.term)

    # Calculate the sum of squares of vector dimensional values
    def getVectorMagnitude(self, term_vector):
        magnitude = 0
        for item in term_vector:
            magnitude += term_vector[item] * term_vector[item]

        return magnitude

    # Find contribution of terms for dot product
    def findMaxWordContribution(self, dotProductNumerator, user_vector_sum_of_squares, target_vector_sum_of_squares,
                                common_stats, source_word_dict):
        index = 0;
        commmon_stat_length = len(common_stats)
        dotProduct = dotProductNumerator / (
                math.sqrt(user_vector_sum_of_squares) * math.sqrt(target_vector_sum_of_squares))
        top_contributors_list_for_arg = []
        while index < commmon_stat_length:
            term = common_stats[index].term
            numerator_contribution_of_term = common_stats[index].model * source_word_dict[term]
            numerator_without_contribution = dotProductNumerator - numerator_contribution_of_term
            user_vector_sum_of_squares_without_contribution = user_vector_sum_of_squares - (
                    source_word_dict[term] * source_word_dict[term])
            target_vector_sum_of_squares_without_contribution = target_vector_sum_of_squares - (
                    common_stats[index].model * common_stats[index].model)

            dot_product_without_contribution = numerator_without_contribution / (
                    math.sqrt(user_vector_sum_of_squares_without_contribution) * math.sqrt(
                target_vector_sum_of_squares_without_contribution))
            term_contribution = dotProduct - dot_product_without_contribution
            top_contributors_len = len(top_contributors_list_for_arg)

            if top_contributors_len < 3:
                top_contributors_list_for_arg.append(TermInfo(term, term_contribution))
                top_contributors_list_for_arg = sorted(top_contributors_list_for_arg,
                                                       key=lambda term_info: term_info.model)
            else:
                if term_contribution >= top_contributors_list_for_arg[0].model:
                    top_contributors_list_for_arg.append(TermInfo(term, term_contribution))
                    top_contributors_list_for_arg = top_contributors_list_for_arg[1:]
                    top_contributors_list_for_arg = sorted(top_contributors_list_for_arg,
                                                           key=lambda term_info: term_info.model)
            index += 1

        return top_contributors_list_for_arg;

    # Calculates the similarity score for a parameter
    def calculateKHighestScores(self, source_word_dict, common_words_dict, uncommon_words_dict, k):
        user_vector_sum_of_squares = self.getVectorMagnitude(source_word_dict);
        user_vector_magnitude = math.sqrt(user_vector_sum_of_squares);
        top_contributors_per_arg = {}
        dot_product_list = []
        for arg in common_words_dict:
            common_stats = common_words_dict[arg];
            uncommon_stats = uncommon_words_dict[arg] if arg in uncommon_words_dict else []
            commmon_stat_length = len(common_stats)
            index = 0;
            vector_magnitude = 0;
            dotProductNumerator = 0;
            top_contributors_list_for_arg = []
            while index < commmon_stat_length:
                vector_magnitude += common_stats[index].model * common_stats[index].model
                numerator_addition = common_stats[index].model * source_word_dict[common_stats[index].term];
                dotProductNumerator += numerator_addition
                index += 1

            for stat in uncommon_stats:
                vector_magnitude += stat * stat;

            top_contributors_list_for_arg = self.findMaxWordContribution(dotProductNumerator,
                                                                         user_vector_sum_of_squares, vector_magnitude,
                                                                         common_stats, source_word_dict);
            vector_magnitude = math.sqrt(vector_magnitude)
            dot_product = dotProductNumerator / (vector_magnitude * user_vector_magnitude)

            dot_product_list_length = len(dot_product_list)
            if dot_product_list_length < k:
                dot_product_list.append(DotProduct(arg, dot_product, top_contributors_list_for_arg))
                dot_product_list = sorted(dot_product_list, key=lambda dot_product_instance: dot_product_instance.value)
            else:
                if dot_product > dot_product_list[0].value:
                    dot_product_list.append(DotProduct(arg, dot_product, top_contributors_list_for_arg))
                    dot_product_list = dot_product_list[1:]
                    dot_product_list = sorted(dot_product_list,
                                              key=lambda dot_product_instance: dot_product_instance.value)

        return dot_product_list;

    # Gets conversion function based on model
    def get_conversion_func(self, model):
        def convert_int(item):
            return int(item)

        return_func = convert_int
        if model == "TF_IDF":
            def convert_float(item):
                return float(item)

            return_func = convert_float

        return return_func

    # Find similar users for given model and user id
    def find_similar_users(self, user_id, model, k):
        get_terms_query = "select term, {1} from termsPerUser where userId = \"{0}\"".format(user_id, model)
        source_word_dict = {};

        get_terms_query_result = self._database_operations.executeSelectQuery(get_terms_query)
        conversion_func = self.get_conversion_func(model)
        for item in get_terms_query_result:
            source_word_dict[item[0]] = conversion_func(item[1])

        join_query_result = "select te.userId,te.term,te.{0}, te1.userId, te1.term, te1.{0} from (select te2.userId, te2.term, te2.{0} from termsPerUser te2 where userId <> \"{1}\")" \
                            " te LEFT JOIN (select userId, term, {0} from termsPerUser where userId = \"{1}\") te1 on te1.term=te.term;".format(
            model, user_id)
        result_set = self._database_operations.executeSelectQuery(join_query_result)
        result = self.process_text_result_sets(result_set, k, source_word_dict, self.get_conversion_func(model))
        self.display_result(result)

    # Find similar images for given model and image id
    def find_similar_images(self, image_id, model, k):
        get_terms_query = "select term, {1} from termsPerImage where imageId = \"{0}\"".format(image_id, model)
        source_word_dict = {};

        get_terms_query_result = self._database_operations.executeSelectQuery(get_terms_query)
        conversion_func = self.get_conversion_func(model)
        for item in get_terms_query_result:
            source_word_dict[item[0]] = conversion_func(item[1])

        join_query_result = "select te.imageId,te.term,te.{0}, te1.imageId, te1.term, te1.{0} from (select te2.imageId, te2.term, te2.{0} from termsPerImage te2 where imageId <> \"{1}\")" \
                            " te LEFT JOIN (select imageId, term, {0} from termsPerImage where imageId = \"{1}\") te1 on te1.term=te.term;".format(
            model, image_id)
        result = self._database_operations.executeSelectQuery(join_query_result)
        result = self.process_text_result_sets(result, k, source_word_dict, self.get_conversion_func(model))
        self.display_result(result)

    # Find similar locations for given model and location id
    def find_similar_locations(self, location_id, model, k):
        get_terms_query = "select term, {1} from termsPerLocation where locationId = \"{0}\"".format(location_id, model)
        source_word_dict = {};

        get_terms_query_result = self._database_operations.executeSelectQuery(get_terms_query)
        conversion_func = self.get_conversion_func(model)
        for item in get_terms_query_result:
            source_word_dict[item[0]] = conversion_func(item[1])

        join_query_result = "select te.locationId,te.term,te.{0}, te1.locationId, te1.term, te1.{0} from (select te2.locationId, te2.term, te2.{0} from termsPerLocation te2 where locationId <> {1})" \
                            " te LEFT JOIN (select locationId, term, {0} from termsPerLocation where locationId = {1}) te1 on te1.term=te.term;".format(
            model, location_id)
        result = self._database_operations.executeSelectQuery(join_query_result)
        result = self.process_text_result_sets(result, k, source_word_dict, self.get_conversion_func(model))
        location_map = DBUtils.create_location_id_key_map(self._database_operations)
        self.display_location_result(result, location_map)
