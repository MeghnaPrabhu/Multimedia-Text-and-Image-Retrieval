# readsd text files line by line
class TextFileProcessor:
    def __init__(self, basePath):
        self.__basePath = basePath

    def yieldLines(self, fileName):
        with open(fileName, encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines:
                yield line;

            f.close();

    def process_text_file(self, fileName, line_processor, helper_object = None):
        queries = []
        for line in self.readFile(fileName):
            queries += line_processor(line, helper_object) if helper_object else line_processor(line)
        return queries;

    def readFile(self, fileName):
        fileName = self.__basePath + fileName;
        return self.yieldLines(fileName);
