from dataclasses import dataclass
from typing import Optional
from PyPDF2 import PdfReader
import numpy as np

@dataclass
class File:
    entity: str
    category: str
    file_data: Optional[bytes] = None
    file_path: Optional[str] = None
    
    def file_name(self) -> str:
        return self.file_path.split('/')[-1]

    def read_and_chunk(self, chunks: int) -> list[str]:
        """
        Reads in an entire file and splits the text into 3 equal sections per page 
        Returns: a list of text sections
        """
        reader = PdfReader(self.file_data if self.file_data else self.file_path)
        text_by_page = [page.extract_text() for page in reader.pages]
        text_by_page_split = []
        prefix = np.array([self.entity, self.category])
        for page in text_by_page:
            # make sure special unicode characters are removed
            page = np.array(page.encode('ascii','ignore').decode().split()) 
            split_string = [
                " ".join(np.concatenate((prefix,section)))
                for section in np.array_split(page,3)
            ]
            text_by_page_split.extend(list(split_string))

        return text_by_page_split
