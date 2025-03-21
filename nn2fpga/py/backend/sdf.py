class Node():
    
    def __init__(self, layer):
      self.name = layer
      self.input_edges = []
      self.output_edges = []
  
    def add_input_edge(self, edge):
      self.input_edges.append(edge)
  
    def add_output_edge(self, edge):
      self.output_edges.append(edge)

class Edge():
    
    def __init__(self, src, dest):
      self.src = src
      self.dest = dest
  
    def set_src(self, src):
      self.src = src
  
    def set_dest(self, dest):
      self.dest = dest

class SDF():
  
  def __init__(self, name):
    self.name = name
    self.nodes = set()
    self.edges = set()

  def add_node(self, layer, input_edges=None, output_edges=None):
    new_node = Node(layer)
   
    if input_edges:
      for edge in input_edges:
        new_node.add_input_edge(edge)

    if output_edges:
      for edge in output_edges:
        edge.set_src(node)
        node.add_output_edge(edge)
    
    self.nodes.append(new_node)
