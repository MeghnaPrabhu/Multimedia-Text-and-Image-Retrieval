from phase1.inputParameter import InputParameter;

class Phase1:
    def __init__(self, query_processor, csv_processor):
        self._query_processor = query_processor
        self._csv_processor = csv_processor
        self._text_options = {1: InputParameter("User Id", query_processor.find_similar_users),
                        2: InputParameter("Image Id", query_processor.find_similar_images),
                        3: InputParameter("Location Id", query_processor.find_similar_locations)
                        }

        self._vis_options = {1: InputParameter("Location Id, Model, k", csv_processor.find_similar_locations_for_given_model),
                       2: InputParameter("Location Id, k", csv_processor.find_similar_locations_for_all_models),
                       }

    def input(self):
        print("Select to search 1. Text Data 2. Visual Data")
        text_or_vis_input = int(input())
        if text_or_vis_input == 1:
            print("Select to search by 1. User Id, 2. Image Id 3.Location Id")
            input_option = int(input())
            input_param = self._text_options[input_option]
            print("Enter {0}, Model and k".format(input_param.primary_param))
            input_str = input();
            splitInput = input_str.split();
            primary_param = splitInput[0]
            model = splitInput[1]
            k = splitInput[2]
            input_param.func(primary_param, model, int(k));

        else:
            print("Select to search by 1. Location for given model, 2. Location for all models")
            input_option = int(input())
            input_param = self._vis_options[input_option]
            print("Enter {0}".format(input_param.primary_param))
            input_str = input();
            splitInput = input_str.split();
            location = int(splitInput[0])
            model = splitInput[1] if input_option == 1 else None
            k = splitInput[2] if input_option == 1 else splitInput[1]

            input_param.func(location, int(k), model) if input_option == 1 else input_param.func(location, int(k))