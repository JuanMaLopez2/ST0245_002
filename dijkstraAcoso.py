import pandas as pd
from queue import PriorityQueue

def inicioProceso():
    # Lectura del csv en dataframe

    df = pd.read_csv("calles_de_medellin_con_acoso.csv", sep=';')

    # Se crea un dataframe con 4 campos
    df1 = df[['origin', 'destination', 'length', 'harassmentRisk']]

    # Dataframe solo con campo origen para mas adelante utilizarlo como los nodos
    df3 = df[['origin']]

    # Cantidad de nodos
    nodos = df3.origin.unique()
    df3 = pd.DataFrame(nodos, columns=['origin'])
    df3.insert(0, 'id', df3.index)
    # print(df3.head())

    #  escribe los nodos en un txt
    with open("Nodos.txt", "w") as text_file:
        txt = str((df3.to_string()))
        text_file.write(txt + "\r\n")

    # Almacena en un dataframe nuevo los destinos de cada nodo origen
    df4 = (df1.loc[df1['origin'].isin(df3["origin"])])
    # print(df4)


    #  escribe los Destinos en un txt
    with open("Destinos.txt", "w") as text_file:
        txt = str((df4.to_string()))
        text_file.write(txt + "\r\n")

    # Se convierte la columna destino con el indice de cada nodo unico
    df4['destination'] = df4['destination'].map(df3.set_index('origin')['id'])
    df4['destination'] = df4['destination'].fillna(0).astype(int)

    # print(df4.head())

    # Se convierte la columna origen con el indice de cada nodo unico
    df4['origin'] = df4['origin'].map(df3.set_index('origin')['id'])
    df4['origin'] = df4['origin'].fillna(0).astype(int)

    # se convierte la columna distancia en float para el dijkstra
    df4['length'] = df4['length'].fillna(0).astype(int)

    # se crea nuevo datframe para el camino con el menor acoso
    dfAcoso = df4[['origin', 'destination', 'harassmentRisk']]

    # dfAcoso['harassmentRisk'] = dfAcoso['harassmentRisk'].apply(lambda x: x*100)
    
    dfAcoso['harassmentRisk'] = dfAcoso['harassmentRisk'].fillna(0)
    # print(dfAcoso.head())

    # Se elimina la columna harassmentRisk ya que para el primer dijkstra solo se necesita la distancia
    df4.drop('harassmentRisk', inplace=True, axis=1)
    # print(df4.head())

    # print((df4.dtypes))

    

    return df1,df3,df4,dfAcoso

def agregar(mylist,dataframetoList,dataframe):
    # Se agregan los nodos
    mylist = dataframetoList["id"].tolist()
    # Se agregan las aristas
    for index, row in dataframe.iterrows():
        add_edge(row['origin'], row['destination'],mylist)

    return mylist

#Funcion que crea una lista adyacente a partir de un dataframe
def add_node(node):
    if node not in mylist:
        mylist.append(node)
    else:
        print("Node ", node, " already exists!")

# agrega las aristas
def add_edge(node1, node2,mylist):
    temp = []
    if node1 in mylist and node2 in mylist:
        if node1 not in adj_list:
            temp.append(node2)
            adj_list[node1] = temp

        elif node1 in adj_list:
            temp.extend(adj_list[node1])
            temp.append(node2)
            adj_list[node1] = temp

#La variable peso puede ser distancia o acoso entre los nodos
def CalculoDistancia(dfNodos,dfDestinos,nombre):
    g = Graph(len(dfNodos))
    # Se agregan las aristas
    print(dfDestinos.columns)

    for i in range(len(dfDestinos)) :
        g.add_edge2(dfDestinos.iloc[i, 0], dfDestinos.iloc[i, 1],dfDestinos.iloc[i, 2])

    D = g.dijkstra(0)
    # print(D)
    # print(type(D))
    texto = ''
    if nombre == 'Acoso':
        texto = 'Acoso desde el nodo 0 al nodo '
    else:
        texto = 'Distancia desde el nodo 0 al nodo '

    # Se imprime la lista adyacente en un txt
    with open("Fin" + nombre + ".txt", "w") as text_file:
        for vertex in range(len(D)):
            txt = (texto, str(vertex), "es", str(D[vertex]) )
            txt = str(txt)
            text_file.write(txt + "\r\n")

def graph(nombre):
    with open("Grafo" + nombre + ".txt", "w") as text_file:
        for node in adj_list:
            txt = (node, " ---> ", [i for i in adj_list[node]])
            txt = str(txt)
            text_file.write(txt + "\r\n")


class Graph:
    def __init__(self, num_of_vertices):
        self.v = num_of_vertices
        self.edges = [[(-1) for i in range(num_of_vertices)]
                    for j in range(num_of_vertices)]
        self.visited = []


    def add_edge2(self, u, v, weight):
        self.edges[u][v] = weight
        self.edges[v][u] = weight

    def dijkstra(self, start_vertex):
        D = {v: float('inf') for v in range(self.v)}
        D[start_vertex] = 0

        pq = PriorityQueue()
        pq.put((0, start_vertex))

        while not pq.empty():
            (dist, current_vertex) = pq.get()
            self.visited.append(current_vertex)

            for neighbor in range(self.v):
                if self.edges[current_vertex][neighbor] != -1:
                    distance = self.edges[current_vertex][neighbor]
                    if neighbor not in self.visited:
                        old_cost = D[neighbor]
                        new_cost = D[current_vertex] + distance
                        if new_cost < old_cost:
                            pq.put((new_cost, neighbor))
                            D[neighbor] = new_cost
        return D

if __name__ == "__main__":
    # df1 dataframe con 4 campos origin,destination,length,harassmentRisk
    # df3 Dataframe solo con campo origen para mas adelante utilizarlo como los nodos
    # df4 Almacena en un dataframe nuevo los destinos de cada nodo origen
    # dfAcoso se crea nuevo datframe para el camino con el menor acoso
    adj_list = {}
    mylist = []
    df1,df3,df4,dfAcoso = inicioProceso()

    # agrego las aristas del grafo
    agregar(mylist,df3,df4)

    # Se imprime el grafo en un txt
    graph('Distancia')

    # Se imprime la lista adyacente en un txt
    with open("ListAdyacente.txt", "w") as text_file:
        txt = str(adj_list)
        text_file.write(txt + "\r\n")
    
    # Calcula la menor distancia desde el nodo inicial al nodo final
    CalculoDistancia(df3,df4,"Distancia")
    
    # Crea el nuevo grafo para calcular la ruta con el menor acoso
    adj_list = {}
    mylist = []
    agregar(mylist,df3,dfAcoso)
    
    graph('Acoso')
    
    # # Se imprime la lista adyacente en un txt
    with open("ListAdyacenteAcoso.txt", "w") as text_file:
        txt = str(adj_list)
        text_file.write(txt + "\r\n")
    
    # Calcula la ruta con el menor acoso
    CalculoDistancia(df3,dfAcoso,"Acoso")