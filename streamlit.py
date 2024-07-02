import streamlit as st
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
 
class TextileGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
 
    def add_node(self, node):
        self.graph.add_node(node)
 
    def remove_node(self, node):
        self.graph.remove_node(node)
 
    def add_edge(self, node1, node2):
        self.graph.add_edge(node1, node2)
 
    def find_shortest_path(self, start_node, end_node):
        try:
            path = nx.shortest_path(self.graph, source=start_node, target=end_node)
            return path
        except nx.NetworkXNoPath:
            return None
 
    def display_graph(self):
        st.write("Nodes:", self.graph.nodes())
        st.write("Edges:", self.graph.edges())
 
 
    def returnNodes(self):
        return list(self.graph.nodes())
 
 
def addCategory(node,parent,graph,model,keywords):
    graph.add_node(node)
    graph.add_edge(parent,node)
    keywords.append(node)
    #embeddingFunction(keywords=keywords,model=model)
    embeddings = model.encode(keywords)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
 
    print(f"node added:{parent}->{node}")
    return index
 
def create_graph(data):
    textile_graph = TextileGraph()
    textile_graph.add_node("Top")
 
    for index, row in data.iterrows():
        level1 = row['Level 1: Categories']
        level2 = row['Level 2: Subcategories']
        level3 = row['Level 3: Detailed Subcategories']
 
        if pd.notna(level1):
            textile_graph.add_node(level1.lower())
            textile_graph.add_edge("Top", level1.lower())
        if pd.notna(level2):
            textile_graph.add_node(level2.lower())
            if pd.notna(level1):
                textile_graph.add_edge(level1.lower(), level2.lower())
        if pd.notna(level3):
            detailed_subcategories = [sub.strip() for sub in level3.split(',')]
            for sub in detailed_subcategories:
                textile_graph.add_node(sub.lower())
                if pd.notna(level2):
                    textile_graph.add_edge(level2.lower(), sub.lower())
 
    return textile_graph
 
def main():
    st.title("Textile Taxonomy Graph")
 
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
 
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
 
        textile_graph = create_graph(data)
        textile_graph.display_graph()
 
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        #index = faiss.read_index("index_file.index")
        #st.write(index)
        #keywords = ['Water Conservation', 'Eco-Friendly Materials', 'Fashion and Apparel', 'Flax', 'Blended Fibers', 'Clothing', 'Upholstery', 'Manufacturing Processes', 'Technical and Industrial Uses', 'Bedding', 'Spandex', 'Agricultural', 'Home and Interior', 'Construction', 'Wool', 'Formal Wear', 'Cotton-Polyester', 'Natural Fibers', 'Waste Reduction', 'Outerwear', 'Curtains', 'Medical', 'Raw Materials', 'Sportswear', 'Open-End Spinning', 'Sustainability', 'Weft Knitting', 'Hemp', 'Industrial Textiles', 'Polyester', 'Digital Printing', 'Plain Weave', 'Screen Printing', 'Worker Rights', 'Non-woven Fabrics', 'Dyeing and Printing', 'Ethical Practices', 'Home Textiles', 'Functional Finishing', 'Wool-Nylon', 'Furniture', 'Knitting', 'Bamboo', 'Nylon', 'Woven Fabrics', 'Silk', 'Air-Jet Spinning', 'Satin Weave', 'Geotextiles', 'Textile Products', 'Acrylic', 'Acid Dyeing', 'Weaving', 'Accessories', 'Energy Efficiency', 'Sustainable Manufacturing Practices', 'Reactive Dyeing', 'Mechanical Finishing', 'Spinning', 'Organic Cotton', 'Twill Weave', 'Ring Spinning', 'Chemical Finishing', 'Fair Trade', 'Protective Textiles', 'Warp Knitting', 'Fabrics', 'Applications', 'Apparel', 'Cotton', 'Polypropylene', 'Medical Textiles', 'Recycled Fibers', 'Knitted Fabrics', 'Casual Wear', 'Decor', 'Towels', 'Synthetic Fibers', 'Automotive', 'Safe Working Conditions', 'Finishing']
 
        keywords = textile_graph.returnNodes()
        print(type(keywords))
        embeddings = model.encode(keywords)
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        #faiss.write_index(index, "index1.index")
 
 
        query_text = st.text_input("Enter Query:")
 
        if query_text:
            query_embedding = model.encode([query_text])
 
            # Number of nearest neighbors to search for
            k = 2
 
            # Perform the search
            distances, indices = index.search(query_embedding, k)
 
            node = keywords[indices[0, 0]].lower()
            path = textile_graph.find_shortest_path("Top", node)
            st.write(f"Path: {path[1]}->{path[2]}->{path[3]}")
 
if __name__ == "__main__":
    main()