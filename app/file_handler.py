from dataclasses import dataclass
from typing import Optional
from pypdf import PdfReader
import numpy as np
from io import BytesIO

@dataclass
class File:
    entity: str
    category: str
    name: Optional[str] = None
    file_data: Optional[bytes] = None
    file_path: Optional[str] = None

    def file_name(self) -> str:
        if result := self.name:
            return result 
        elif result := self.file_path:
            return result.split('/')[-1]
        else:
            return f"{self.entity}_{self.category}.pdf"
    
    def read_file(self) -> None:
        self.reader = PdfReader(BytesIO(self.file_data) if self.file_data else self.file_path)
        self.num_pages = len(self.reader.pages)

    def read_and_chunk(self, chunks: int) -> list[str]:
        """
        Reads in an entire file and splits the text into equal sections (chunks) per page 
        Returns: a list of text sections
        """
        self.read_file()
        text_by_page = [page.extract_text() for page in self.reader.pages]
        text_by_page_split = []
        prefix = np.array([self.entity, self.category])
        for page in text_by_page:
            # make sure special unicode characters are removed
            page = np.array(page.encode('ascii','ignore').decode().split()) 
            split_string = [
                " ".join(np.concatenate((prefix,section)))
                for section in np.array_split(page,chunks)
            ]
            text_by_page_split.extend(list(split_string))

        return text_by_page_split

    def add_embedding(self, embedding: list[float]) -> None:
        self.embedding = embedding