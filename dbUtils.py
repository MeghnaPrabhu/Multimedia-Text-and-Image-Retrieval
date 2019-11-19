class DBUtils:
    # Contains functions to get lo_id to loc_key map and vice versa
    @staticmethod
    def create_location_key_id_map(database_operations):
        result = database_operations.executeSelectQuery("select locationKey, locationId from locations")
        location_key_id_map = {}
        for row in result:
            location_key_id_map[row[0]] = row[1]
        return location_key_id_map

    @staticmethod
    def create_location_id_key_map(database_operations):
        result = database_operations.executeSelectQuery("select locationKey, locationId from locations")
        location_key_id_map = {}
        for row in result:
            location_key_id_map[row[1]] = row[0]
        return location_key_id_map
