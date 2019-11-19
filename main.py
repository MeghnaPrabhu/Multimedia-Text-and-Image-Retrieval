import sys
from pathlib import Path

from databaseOperations import DatabaseOperations;
from initialSetup import InitialSetup;
from input_by_phase.phase1 import Phase1
from input_by_phase.phase2.phase2 import Phase2
from input_by_phase.phase3.phase3 import Phase3
from phase1.csvProcessor import CsvProcessor
from phase1.queryProcessor import QueryProcessor;

base_file_path = sys.argv[1]
#  if sys.argv[1] else "/home/vivemeno/MWD Project Data/devset/extractedData";
# sqlfilePath =  "/home/vivemeno/MWD Project Data/database/test7.db";
sqlfilePath = base_file_path + "/test7.db"
base_path = base_file_path
suffix_image_dir = "/descvis/img"
model="CM"
my_file = Path(sqlfilePath)
database_operations = DatabaseOperations(sqlfilePath)
# checks if file is present
if not my_file.is_file():
    initialSetup = InitialSetup(database_operations, base_path)
    initialSetup.setup_database_from_devset_data()

query_processor = QueryProcessor(database_operations)
csv_processor = CsvProcessor(base_file_path + suffix_image_dir, database_operations)

phase_options = {1: Phase1(query_processor, csv_processor),
                 2: Phase2(base_path + suffix_image_dir, database_operations),
                 3: Phase3(base_path, database_operations)}

keep_running = True
while keep_running:
    print("Select Phase 1. PhaseI 2. PhaseII 3. Phase III :")
    phase_option = int(input())
    phase = phase_options[phase_option]
    phase.input()

    print("Press N to stop")
    input_option = input()
    if input_option == "N":
        keep_running = False
