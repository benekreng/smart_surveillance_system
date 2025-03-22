import os
import numpy as np

class FaceStore:
    def __init__(self, db_dir):
        self.embeddings_dir = db_dir
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)
        
        self.max_index = 0
        self.stored_faces = []
        self.mappings = []

        # Load embeddings into memory
        for idx, filename in enumerate(os.listdir(self.embeddings_dir)):
            f = os.path.join(self.embeddings_dir, filename)
            if os.path.isfile(f):
                # Load face embedding
                embedding = np.load(f)
                self.stored_faces.append(embedding)
                # Exctract face embeding and index form name
                index, name = int(filename.split("_")[0]), filename.split("_")[1]
                
                # Create in memory object of embedding name and indexes
                self.max_index = max(self.max_index, index)
                self.mappings.append([index, name.split(".")[0]])

    def __setitem__(self, key, value):
        self._store_embedding(key, value)

    def _store_embedding(self, name, embedding):
        if name in [i[1] for i in self.mappings]:
            raise ValueError("Name already in Database")
        # Increment index
        self.max_index = self.max_index+1
        # Name file
        filename = f"{self.max_index}_{name}"
        file_path = f"{self.embeddings_dir}{filename}.npy"

        # Save file to disk and store in memory 
        np.save(file_path, embedding)
        self.stored_faces.append(embedding)
        self.mappings.append([self.max_index, name])
        self._sort_mappings()
    
    def _sort_mappings(self):
        if len(self.mappings) > 0:
            self.mappings = sorted(self.mappings, key=lambda x: x[0])

    def remove_identity(self, name):
        # Remove an identity from the database
        indexes_to_remove = [i for i, (_, stored_name) in enumerate(self.mappings) if stored_name == name]

        if not indexes_to_remove:
            raise ValueError("Identity not found in database")

        # Remove the corresponding embeddings and mapping
        for index in sorted(indexes_to_remove, reverse=True):
            del self.stored_faces[index]
            del self.mappings[index]

        # Delete the stored embedding file
        for file in os.listdir(self.embeddings_dir):
            if file.endswith(".npy") and name in file:
                os.remove(os.path.join(self.embeddings_dir, file))

        # Sort mappings after deletion
        self._sort_mappings()

        