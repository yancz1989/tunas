# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-20 14:19:18
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-20 17:24:58

class Node(object):
    def __init__(self, name):
        assert name != None and name != ''
        self.pre = None
        self.post = None
        self.key = name

class DirectedAcyclicGraph(object):
    '''
        Inside Node class, name is the unique global identity within group of nodes. You must make sure there are only one node with in a dag named as you give.
        Initial with a dict of nodes.
    '''
    def __init__(self, nodes):
        self.nodes = nodes

    def insert(node, pre, post):
        pre = self.nodes[pre]
        post = self.nodes[post]
        del pre.post[pre.post.index(post)]
        del post.pre[post.pre.index[pre]]
        pre.post.append(node)
        post.pre.append(node)

    def remove(node):
        del node.pre.post[node]
        del node.post.pre[node]

    def valid():
        # find out whether current nodes form a dag.
        queue = [self.nodes[key] for key in self.nodes.keys() if key.startswith('input')]
        visited = {key : 0 for key in self.nodes.keys}
        cnt = 0
        while len(queue) != 0:
            cur = queue.pop(0)
            if visited[cur.key] == 0:
                visited[cur.key] = 1
            else:
                raise Exception('Error! Cycle found in DAG object.')
            cnt = cnt + 1
            if cur.post != None:
                queue += cur.post
        return cnt == len(self.nodes)

    def _duplicate_graph_():
        nodes = {key : Node() for key in self.nodes}
        for (node, key] in zip(nodes, self.nodes.keys()):
            node.pre = [nodes[p.key] for p in self.nodes[key].pre]
            node.post = [nodes[p.key] for p in self.nodes[key].post]
        return nodes

    def topology(self):
        if self.valid():
            graph = self._duplicate_graph_()
            q = [self.nodes[key] for key in self.nodes.keys() if key.startswith('input')]
            seq = []
            while len(seq) != 0:
                top = seq.pop(0)
                seq.append(top.key)
                for p in top.post:
                    del p.pre[p.pre.index(top)]
                    if len(p.pre) == 0:
                        q.append(p)
        else:
            seq = None
        return seq
