from games.starcraft2.vector_database.Chroma_vdb import ChromaDBManager
from games.starcraft2.embedding.embedding_model import LocalEmbedding
from games.starcraft2.utils.action_info import ActionDescriptions
from games.starcraft2.utils.action_extractor import *
import json
import os

class ActionDBManager:
    def __init__(self, db_path):
        self.manager = ChromaDBManager(db_path)
        self.embedding_model = LocalEmbedding()
        self.db_path = db_path

    def initialize_collection(self, collection_name, similarity_type="cosine"):
        self.manager.create_or_get_collection(name=collection_name, similarity_type=similarity_type)
        print(f"{collection_name} Collection created or retrieved.")
        print("vdb_path: ", self.db_path)

    def add_actions_to_db(self, actions_list):
        docs = [{"id": str(i), "content": action, "metadata": {"timestamp": 0.0}}
                for i, action in enumerate(actions_list)]
        embeddings = [self.embedding_model.embed_text(doc["content"]) for doc in docs]
        self.manager.add_documents(documents=docs, embeddings=embeddings)
        print("Documents added.")

    def search_actions(self, query, top_k=5):
        """
        Search for semantically similar actions using the given query string
        :param query: Query string
        :param top_k: Return the top k most similar results
        :return: List of search results
        """
        query_embedding = self.embedding_model.embed_text(query)
        query_result = self.manager.query(embeddings=query_embedding, n_results=top_k)

        # Process query results
        formatted_result = {
            'ids': query_result['ids'][0] if query_result['ids'] else [],
            'documents': query_result['documents'][0] if query_result['documents'] else [],
            'metadatas': query_result['metadatas'][0] if query_result['metadatas'] else [],
            'embeddings': query_result.get('embeddings', [])
        }

        return formatted_result

    def populate_action_db(self, collection_name, race, similarity_type="cosine"):
        """
        Initialize the collection and add the list of actions to the database
        :param collection_name: Name of the collection to create or retrieve
        :param race: Specified race, e.g., "Protoss" or "Zerg"
        :param similarity_type: Similarity metric type for the collection, default is "cosine"
        """
        self.initialize_collection(collection_name, similarity_type)
        action_desc = ActionDescriptions(race)
        actions_list = list(action_desc.flattened_actions.values())
        self.add_actions_to_db(actions_list)


def interactive_search(action_db_manager):
    """
    Search using a query string input by the user and print the search results
    :param action_db_manager: Instance of ActionDBManager
    """
    while True:
        query = input("Please enter the action you are searching for (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        search_results = action_db_manager.search_actions(query)
        print("Search results:", search_results)


def read_command(command_file):
    with open(command_file, 'r', encoding='utf-8') as file:
        command_data = json.load(file)
    return command_data


def extract_and_search_actions(command_file, db_path):
    # Initialize ActionDBManager object
    action_db_manager = ActionDBManager(db_path)
    action_db_manager.initialize_collection("protoss_actions")

    # Read commands
    commands = read_command(command_file)

    action_desc = ActionDescriptions("Protoss")  # Read action descriptions
    action_dict = action_desc.action_descriptions  # Read action dictionary
    action_extractor = ActionExtractor(action_dict)  # Initialize action extractor
    empty_idx = action_desc.empty_action_id  # Index of empty action

    for i in range(len(commands)):
        command = commands[i]
        if isinstance(command, list):
            command = " ".join(command)

        # Extract actions based on the modified function
        action_ids, valid_actions = extract_actions_from_command(command, action_extractor=action_extractor,
                                                                 empty_idx=empty_idx,
                                                                 action_db_manager=action_db_manager)

        # Output counter and related information for debugging
        print("Valid actions found: ", valid_actions)
        print("Action IDs: ", action_ids)
        for id in action_ids:
            if type(id) != int:
                raise ValueError(f"Action ID must be an integer, but found ID of type {type(id)}.")

        if not valid_actions:
            print(f"Extraction Problem at Command Counter: {i}")


def configure_protoss_actions(db_path):
    action_db_manager = ActionDBManager(db_path)
    action_db_manager.populate_action_db("protoss_actions", "Protoss")


def configure_terran_actions(db_path):
    action_db_manager = ActionDBManager(db_path)
    action_db_manager.populate_action_db("terran_actions", "Terran")


def configure_zerg_actions(db_path):
    action_db_manager = ActionDBManager(db_path)
    action_db_manager.populate_action_db("zerg_actions", "Zerg")




if __name__ == "__main__":
    relative_path_parts = ["..", "..", "utils", "actionvdb", "action_vdb"]
    db_path = os.path.join(*relative_path_parts)
    configure_zerg_actions(db_path)
    configure_protoss_actions(db_path)
