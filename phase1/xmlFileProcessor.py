import xml.etree.ElementTree as ET

class XmlFileProcessor:

    def __init__(self, base_path):
        self.__basePath = base_path

    def parse_xml(self, file, file_processor):
        tree = ET.parse(self.__basePath + file)
        root = tree.getroot()
        file_processor(root)
