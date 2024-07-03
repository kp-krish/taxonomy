import streamlit as st
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
from pyvis.network import Network
import streamlit.components.v1 as components

MAX_FILE_SIZE_MB = 2
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB*1024*1024

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

    def display_graph_pyvis(self):
        net = Network(notebook=True)
        net.from_nx(self.graph)
        path = "network.html"
        net.save_graph(path)
        HtmlFile = open(path, 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height=500)
 
    def returnNodes(self):
        return list(self.graph.nodes())

def addCategory(node,parent,graph,model,keywords):
    graph.add_node(node.lower())
    graph.add_edge(parent.lower(),node)
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
    textile_graph.add_node("top")
    clms=data.columns.tolist()
    print(clms)
    for index, row in data.iterrows():
        level1 = row[f'{clms[0]}']
        level2 = row[f'{clms[1]}']
        level3 = row[f'{clms[2]}']
 
        if pd.notna(level1):
            textile_graph.add_node(level1.lower())
            textile_graph.add_edge("top", level1.lower())
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

def genOutput(keywords, textile_graph, index, model, i):
    st.write("Press 1: To get Search product/type:")
    st.write("Press 2: To add new product/type")
    st.write("Press 3: To display graph")
    i = i + 1
    choice = st.text_input("Enter choice:", key=f"c_{i}")
    if choice:
        choice = choice.strip()
        if choice == "1":
            i = i + 1
            query_text = st.text_input("Enter product that you want to search", key=f"c_{i}")
            if query_text:
                query_embedding = model.encode([query_text])
                k = 2
                distances, indices = index.search(query_embedding, k)
                node = keywords[indices[0, 0]].lower()
                path = textile_graph.find_shortest_path("Top", node)
                st.write(f"Path: {path[1:]}")
                genOutput(keywords, textile_graph, index, model, i)
        elif choice == "2":
            i += 1
            node = st.text_input("Enter the new type/product that you want to add", key=f"c_{i}")
            if node:
                i += 1
                parent = st.text_input("Enter the parent type for this new product", key=f"c_{i}")
                if parent:
                    parent = parent.lower()
                    if (parent) not in keywords:
                        st.error("No such parent exist in graph")
                    else:
                        keywords.append(node)
                        embeddings = model.encode(keywords)
                        d = embeddings.shape[1]
                        index = faiss.IndexFlatL2(d)
                        index.add(embeddings)
                        addCategory(node=node, parent=parent, graph=textile_graph, model=model, keywords=keywords)
                        genOutput(keywords, textile_graph, index, model, i)
        elif choice == "3":
            textile_graph.display_graph_pyvis()
            genOutput(keywords, textile_graph, index, model, i)
        else:
            st.error("Please enter valid Choice")
            genOutput(keywords, textile_graph, index, model, i) 


def main():
    st.title("Taxonomy Graph")
 
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        file_size = uploaded_file.size
        if file_size > MAX_FILE_SIZE_BYTES:
            st.error(f"The file size should not exceed {MAX_FILE_SIZE_MB} MB. Please upload a smaller file.")
        else:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
    
            textile_graph = create_graph(data)
            # textile_graph.display_graph()
            print(type(textile_graph))
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            #index = faiss.read_index("index_file.index")
            #st.write(index)
            #keywords = ['Water Conservation', 'Eco-Friendly Materials', 'Fashion and Apparel', 'Flax', 'Blended Fibers', 'Clothing', 'Upholstery', 'Manufacturing Processes', 'Technical and Industrial Uses', 'Bedding', 'Spandex', 'Agricultural', 'Home and Interior', 'Construction', 'Wool', 'Formal Wear', 'Cotton-Polyester', 'Natural Fibers', 'Waste Reduction', 'Outerwear', 'Curtains', 'Medical', 'Raw Materials', 'Sportswear', 'Open-End Spinning', 'Sustainability', 'Weft Knitting', 'Hemp', 'Industrial Textiles', 'Polyester', 'Digital Printing', 'Plain Weave', 'Screen Printing', 'Worker Rights', 'Non-woven Fabrics', 'Dyeing and Printing', 'Ethical Practices', 'Home Textiles', 'Functional Finishing', 'Wool-Nylon', 'Furniture', 'Knitting', 'Bamboo', 'Nylon', 'Woven Fabrics', 'Silk', 'Air-Jet Spinning', 'Satin Weave', 'Geotextiles', 'Textile Products', 'Acrylic', 'Acid Dyeing', 'Weaving', 'Accessories', 'Energy Efficiency', 'Sustainable Manufacturing Practices', 'Reactive Dyeing', 'Mechanical Finishing', 'Spinning', 'Organic Cotton', 'Twill Weave', 'Ring Spinning', 'Chemical Finishing', 'Fair Trade', 'Protective Textiles', 'Warp Knitting', 'Fabrics', 'Applications', 'Apparel', 'Cotton', 'Polypropylene', 'Medical Textiles', 'Recycled Fibers', 'Knitted Fabrics', 'Casual Wear', 'Decor', 'Towels', 'Synthetic Fibers', 'Automotive', 'Safe Working Conditions', 'Finishing']
    
            keywords = textile_graph.returnNodes()
            embeddings = model.encode(keywords)
            d = embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embeddings)
            #faiss.write_index(index, "index1.index")
            genOutput(keywords, textile_graph, index, model, 0)
    
if __name__ == "__main__":
    main()
