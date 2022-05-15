import pandas as pd
from queue import PriorityQueue
from collections import namedtuple, deque
from pprint import pprint as pp
import networkx as nx
from regex import P
import gmplot # libreria para graficar el mapa

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

    # se convierte la columna distancia en float para el dijkstra
    df4['length'] = df4['length'].fillna(0).astype(float)

    # se crea nuevo datframe para el camino con el menor acoso
    dfAcoso = df4[['origin', 'destination', 'length', 'harassmentRisk']]

    # se llenan los vacios con el promedio
    dfAcoso['harassmentRisk'] = dfAcoso['harassmentRisk'].fillna(
        df["harassmentRisk"].mean())
    print(dfAcoso.tail())

    # se agrega una columna nueva donde se multiplica el acoso por la distancia
    dfAcoso['ponderado'] = dfAcoso['length'] * dfAcoso['harassmentRisk']

    print(dfAcoso.head())

    # Se elimina la columna harassmentRisk ya que para el primer dijkstra solo se necesita la distancia
    df4.drop('harassmentRisk', inplace=True, axis=1)

    return df1, df3, df4, dfAcoso

def CalculoDistancia(df, inicio, fin, nombre):
    # se agrega la columna test con la palabra weight para proximamente crear una columna nueva
    # la columna nueva se llamará merged con la siguiente estructura {'weight': 42.867}
    df.insert(loc=3,
              column='test',
              value='weight')
    
    # aca se genera la columna nueva merged dependiendo el proceso
    # si es con el ponderad (distancia*acoso) o con la distancia
    if nombre == 'Acoso':
        df['merged'] = df.apply(lambda row: {row['test']: row['ponderado']}, axis=1)
    else:
        df['merged'] = df.apply(lambda row: {row['test']: row['length']}, axis=1)

    print(df.head())

    # el nuevo dataframe se convierte en una lista para crear el grafo con la libreria networkx
    df1 = df[['origin', 'destination', 'merged']]
    records = df1.to_records(index=False)
    edges = list(records)

    # se crea el objeto grafo
    G = nx.Graph()

    # agrego los nodos unicos del grafo
    for i in range(len(df)):
        G.add_node(df["origin"][i])

    # agrego los bordes/aristas del grafo
    G.add_edges_from(edges)

    print(G)

    # Retorna el camino más corto en coordenadas, desde el inicio hasta el fin ingresado
    # Dependiendo si es solo la distancia o la distancia con menor acoso
    camino = nx.shortest_path(G, source=inicio, target=fin, weight='weight')

    # Retorna el valor de la distancia recorrida usando el camino mas corto
    # Dependiendo si es solo la distancia o la distancia con menor acoso
    length = nx.shortest_path_length(
        G, source=inicio, target=fin, weight='weight')

    texto = ''
    if nombre == 'Acoso':
        texto = 'Distancia recorrida con menor riesgo de acoso: ' + str(length)
    else:
        texto = 'Distancia recorrida: ' + str(length)

    print('El camino más corto desde el nodo ' + str(inicio) +
          ' hasta el nodo ' + str(fin) + ' es: ' + str(camino))
    print(texto)

    return camino

class Graph:
    def __init__(self, edges):
        self.edges = [Edge(*edge) for edge in edges]
        self.vertices = {e.start for e in self.edges} | {
            e.end for e in self.edges}

    # algoritmo de dijkstra
    def dijkstra(self, source, dest):
        assert source in self.vertices
        dist = {vertex: inf for vertex in self.vertices}
        previous = {vertex: None for vertex in self.vertices}
        dist[source] = 0
        q = self.vertices.copy()
        neighbours = {vertex: set() for vertex in self.vertices}
        for start, end, cost in self.edges:
            neighbours[start].add((end, cost))
            neighbours[end].add((start, cost))

        while q:
            u = min(q, key=lambda vertex: dist[vertex])
            q.remove(u)
            if dist[u] == inf or u == dest:
                break
            for v, cost in neighbours[u]:
                alt = dist[u] + cost
                if alt < dist[v]:
                    dist[v] = alt
                    previous[v] = u
        s, u = deque(), dest
        while previous[u]:
            s.appendleft(u)
            u = previous[u]
        s.appendleft(u)
        return s


def dibujarMapa(origen_lat,origen_long,destino_lat,destino_long,camino,proceso):
    # Crea el plotter para mapa
    gmap = gmplot.GoogleMapPlotter(origen_lat,origen_long, 14)

    # Marca el punto de origen en el mapa
    gmap.marker(origen_lat,origen_long, color='red')

    # Resalta en el mapa los nodos por los que va el camino mas corto
    nodos_latitud, nodos_longitud = zip(*camino)
    gmap.scatter(nodos_latitud, nodos_longitud, color='yellow', size=5, marker=False)

    # grafica el poligono y la ruta mas corta en el mapa
    poligono = zip(*camino)
    gmap.polygon(*poligono, color='cornflowerblue', edge_width=10)

    # Marca el punto destino en el mapa
    gmap.marker(destino_lat,destino_long, color='green')

    # Genera un html con el mapa
    if proceso == 'Acoso':
        gmap.draw('mapAcoso.html')
    else:
        gmap.draw('mapDistancia.html')

def transfCamino(camino):
    # se invirten las coordenadas para poder graficarlas usando maps de google
    # en maps primero se usa la latitud (6.200) y despues la longitud -75.5700
    dfRuta = pd.DataFrame (camino, columns = ['ruta'])
    print(dfRuta.head())
    dfRuta['ruta']=( dfRuta.ruta.str.split()
              .apply(lambda x: ', '.join(x[::-1]).rstrip(','))
              .where(dfRuta['ruta'].str.contains(','),dfRuta['ruta']) )

    dfRuta['ruta'] = dfRuta['ruta'].str.replace(')','')
    dfRuta['ruta'] = dfRuta['ruta'].str.replace('(','')
    dfRuta['ruta'] = dfRuta['ruta'].str.replace(' ','')

    print(dfRuta.head())

    # se genera la lista con los valores para graficar el mapa
    foo = lambda x: pd.Series([i for i in reversed(x.split(','))])
    rev = dfRuta['ruta'].apply(foo)
    rev = rev.astype(float)
    rev2 = rev[[1,0]]

    # se convierte en una lista de tuplas para poder ser leido por la funcion y graficarlo
    # ejemplo (6.338773,-75.6909483),(6.338773,-75.6909483),(6.338773,-75.6909483)
    records = rev2.to_records(index=False)
    result_camino = list(records)

    return result_camino

if __name__ == "__main__":
    inf = float('inf')
    Edge = namedtuple('Edge', ['start', 'end', 'cost'])

    # df1 dataframe con 4 campos origin,destination,length,harassmentRisk
    # df3 Dataframe solo con campo origen para mas adelante utilizarlo como los nodos
    # df4 Almacena en un dataframe nuevo los destinos de cada nodo origen
    # dfAcoso se crea nuevo datframe para el camino con el menor acoso

    adj_list = {}
    mylist = []
    df1, df3, df4, dfAcoso = inicioProceso()

    origen_x,origen_y = input("Ingrese las coordenadas de origen: ").split(',')
    destino_x,destino_y = input("Ingrese las coordenadas destino: ").split(',')

    origen= ("(" + str(origen_x) + ', ' + str(origen_y) + ")")
    destino= ("(" + str(destino_x) +', ' + str(destino_y) + ")")

    # Este es para calcular solo el de menor distancia
    camino = CalculoDistancia(df4, origen, destino, 'Distancia')
   
    # funcion para invertir las coordenadas y convertirlas en una lista de tuplas
    result_camino = transfCamino(camino)

    # se envian las cordenadas del origen, del destino y el camino arrojado por el dijkstra
    dibujarMapa(float(origen_y),float(origen_x),float(destino_y),float(destino_x),result_camino,'Distancia')

    # Este es para calcular el de menor distancia y con menos riesgo
    caminoAcoso = CalculoDistancia(dfAcoso, origen, destino, 'Acoso')

     # funcion para invertir las coordenadas y convertirlas en una lista de tuplas
    result_camino = transfCamino(caminoAcoso)

    # se envian las cordenadas del origen, del destino y el camino arrojado por el dijkstra
    dibujarMapa(float(origen_y),float(origen_x),float(destino_y),float(destino_x),result_camino,'Acoso')



