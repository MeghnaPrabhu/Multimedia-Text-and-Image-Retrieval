class LocationInfo:
    def __init__(self, loc_id, loc_key, latitude, longitude, wiki):
        self._loc_id = loc_id
        self._loc_key = loc_key
        self._latitude = latitude
        self._longitude = longitude
        self._wiki = wiki

    @property
    def loc_key(self):
        return self._loc_id

    @property
    def loc_id(self):
        return self._loc_id

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self.longitude

    @property
    def wiki(self):
        return self._wiki
