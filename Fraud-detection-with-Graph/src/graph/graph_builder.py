"""
Graph construction module for fraud detection.
Builds heterogeneous graphs from transaction data with multiple entity types.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from loguru import logger


@dataclass
class Entity:
    """Base class for graph entities"""
    id: str
    entity_type: str
    attributes: Dict[str, Any]
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Transaction:
    """Transaction entity"""
    transaction_id: str
    card_id: str
    merchant_id: str
    amount: float
    timestamp: datetime
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    location: Optional[Tuple[float, float]] = None
    is_fraud: bool = False
    additional_features: Dict[str, Any] = None


class FraudGraph:
    """
    Heterogeneous graph for fraud detection.
    Supports multiple node types and edge relationships.
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_types = set()
        self.edge_types = set()
        self.entity_counts = {}
        
    def add_entity(self, entity: Entity):
        """Add an entity node to the graph"""
        self.graph.add_node(
            entity.id,
            entity_type=entity.entity_type,
            **entity.attributes
        )
        self.node_types.add(entity.entity_type)
        self.entity_counts[entity.entity_type] = self.entity_counts.get(entity.entity_type, 0) + 1
        
    def add_relationship(self, source_id: str, target_id: str, 
                        relationship_type: str, attributes: Dict[str, Any] = None):
        """Add a relationship edge between entities"""
        attributes = attributes or {}
        self.graph.add_edge(
            source_id, target_id,
            relationship_type=relationship_type,
            **attributes
        )
        self.edge_types.add(relationship_type)
        
    def get_subgraph(self, node_ids: List[str], max_depth: int = 2) -> nx.MultiDiGraph:
        """Extract subgraph around specified nodes"""
        nodes_to_include = set(node_ids)
        
        for _ in range(max_depth):
            new_nodes = set()
            for node in nodes_to_include:
                if node in self.graph:
                    new_nodes.update(self.graph.neighbors(node))
                    new_nodes.update(self.graph.predecessors(node))
            nodes_to_include.update(new_nodes)
            
        return self.graph.subgraph(nodes_to_include)
    
    def get_node_features(self, node_id: str) -> Dict[str, Any]:
        """Get all features for a node"""
        if node_id not in self.graph:
            return {}
        return dict(self.graph.nodes[node_id])
    
    def get_edge_features(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """Get all edge features between two nodes"""
        if not self.graph.has_edge(source_id, target_id):
            return []
        
        edges = []
        for key in self.graph[source_id][target_id]:
            edge_data = dict(self.graph[source_id][target_id][key])
            edges.append(edge_data)
        return edges


class GraphBuilder:
    """
    Builds fraud detection graphs from transaction data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.fraud_graph = FraudGraph()
        self.transaction_cache = {}
        
    def build_from_transactions(self, transactions: List[Transaction]) -> FraudGraph:
        """Build graph from transaction list"""
        logger.info(f"Building graph from {len(transactions)} transactions")
        
        # Sort transactions by timestamp
        transactions.sort(key=lambda x: x.timestamp)
        
        for transaction in transactions:
            self._process_transaction(transaction)
            
        self._add_temporal_edges(transactions)
        self._add_behavioral_patterns()
        self._detect_velocity_patterns()
        
        logger.info(f"Graph built with {self.fraud_graph.graph.number_of_nodes()} nodes "
                   f"and {self.fraud_graph.graph.number_of_edges()} edges")
        
        return self.fraud_graph
    
    def _process_transaction(self, transaction: Transaction):
        """Process a single transaction and add to graph"""
        # Add transaction node
        self.fraud_graph.add_entity(Entity(
            id=f"txn_{transaction.transaction_id}",
            entity_type="transaction",
            attributes={
                "amount": transaction.amount,
                "timestamp": transaction.timestamp.isoformat(),
                "is_fraud": transaction.is_fraud,
                **(transaction.additional_features or {})
            }
        ))
        
        # Add card node
        card_id = f"card_{transaction.card_id}"
        if card_id not in self.fraud_graph.graph:
            self.fraud_graph.add_entity(Entity(
                id=card_id,
                entity_type="card",
                attributes={"card_number": transaction.card_id}
            ))
        
        # Add merchant node
        merchant_id = f"merchant_{transaction.merchant_id}"
        if merchant_id not in self.fraud_graph.graph:
            self.fraud_graph.add_entity(Entity(
                id=merchant_id,
                entity_type="merchant",
                attributes={"merchant_name": transaction.merchant_id}
            ))
        
        # Add device node if available
        if transaction.device_id:
            device_id = f"device_{transaction.device_id}"
            if device_id not in self.fraud_graph.graph:
                self.fraud_graph.add_entity(Entity(
                    id=device_id,
                    entity_type="device",
                    attributes={"device_fingerprint": transaction.device_id}
                ))
            self.fraud_graph.add_relationship(
                card_id, device_id, "used_device",
                {"timestamp": transaction.timestamp.isoformat()}
            )
        
        # Add IP node if available
        if transaction.ip_address:
            ip_id = f"ip_{transaction.ip_address}"
            if ip_id not in self.fraud_graph.graph:
                self.fraud_graph.add_entity(Entity(
                    id=ip_id,
                    entity_type="ip_address",
                    attributes={
                        "ip": transaction.ip_address,
                        "location": transaction.location
                    }
                ))
            self.fraud_graph.add_relationship(
                card_id, ip_id, "connected_from",
                {"timestamp": transaction.timestamp.isoformat()}
            )
        
        # Add relationships
        self.fraud_graph.add_relationship(
            card_id, f"txn_{transaction.transaction_id}", "made_transaction",
            {"amount": transaction.amount, "timestamp": transaction.timestamp.isoformat()}
        )
        
        self.fraud_graph.add_relationship(
            f"txn_{transaction.transaction_id}", merchant_id, "at_merchant",
            {"amount": transaction.amount, "timestamp": transaction.timestamp.isoformat()}
        )
        
        # Cache transaction for temporal analysis
        self.transaction_cache[transaction.transaction_id] = transaction
    
    def _add_temporal_edges(self, transactions: List[Transaction]):
        """Add temporal relationships between transactions"""
        time_window = timedelta(hours=self.config.get("temporal_window_hours", 24))
        
        # Group transactions by card
        card_transactions = {}
        for txn in transactions:
            card_id = txn.card_id
            if card_id not in card_transactions:
                card_transactions[card_id] = []
            card_transactions[card_id].append(txn)
        
        # Add temporal edges
        for card_id, txns in card_transactions.items():
            txns.sort(key=lambda x: x.timestamp)
            for i in range(len(txns) - 1):
                if txns[i + 1].timestamp - txns[i].timestamp <= time_window:
                    self.fraud_graph.add_relationship(
                        f"txn_{txns[i].transaction_id}",
                        f"txn_{txns[i + 1].transaction_id}",
                        "followed_by",
                        {
                            "time_diff_seconds": (txns[i + 1].timestamp - txns[i].timestamp).total_seconds(),
                            "same_card": True
                        }
                    )
    
    def _add_behavioral_patterns(self):
        """Detect and add behavioral pattern edges"""
        # Find cards that share devices or IPs
        device_to_cards = {}
        ip_to_cards = {}
        
        for node_id, data in self.fraud_graph.graph.nodes(data=True):
            if data.get("entity_type") == "card":
                # Find connected devices
                for neighbor in self.fraud_graph.graph.neighbors(node_id):
                    neighbor_data = self.fraud_graph.graph.nodes[neighbor]
                    
                    if neighbor_data.get("entity_type") == "device":
                        device_id = neighbor
                        if device_id not in device_to_cards:
                            device_to_cards[device_id] = []
                        device_to_cards[device_id].append(node_id)
                    
                    elif neighbor_data.get("entity_type") == "ip_address":
                        ip_id = neighbor
                        if ip_id not in ip_to_cards:
                            ip_to_cards[ip_id] = []
                        ip_to_cards[ip_id].append(node_id)
        
        # Add shared device relationships
        for device_id, cards in device_to_cards.items():
            if len(cards) > 1:
                for i in range(len(cards)):
                    for j in range(i + 1, len(cards)):
                        self.fraud_graph.add_relationship(
                            cards[i], cards[j], "shares_device",
                            {"device_id": device_id, "risk_score": len(cards) / 10.0}
                        )
        
        # Add shared IP relationships
        for ip_id, cards in ip_to_cards.items():
            if len(cards) > 1:
                for i in range(len(cards)):
                    for j in range(i + 1, len(cards)):
                        self.fraud_graph.add_relationship(
                            cards[i], cards[j], "shares_ip",
                            {"ip_id": ip_id, "risk_score": len(cards) / 20.0}
                        )
    
    def _detect_velocity_patterns(self):
        """Detect high-velocity transaction patterns"""
        velocity_threshold = self.config.get("velocity_threshold", 5)
        time_window = timedelta(hours=self.config.get("velocity_window_hours", 1))
        
        # Group transactions by card and time windows
        for node_id, data in self.fraud_graph.graph.nodes(data=True):
            if data.get("entity_type") == "card":
                transactions = []
                
                # Get all transactions for this card
                for neighbor in self.fraud_graph.graph.neighbors(node_id):
                    neighbor_data = self.fraud_graph.graph.nodes[neighbor]
                    if neighbor_data.get("entity_type") == "transaction":
                        timestamp = datetime.fromisoformat(neighbor_data["timestamp"])
                        transactions.append((neighbor, timestamp))
                
                # Sort by timestamp
                transactions.sort(key=lambda x: x[1])
                
                # Check for velocity patterns
                for i in range(len(transactions)):
                    count = 1
                    j = i + 1
                    
                    while j < len(transactions) and transactions[j][1] - transactions[i][1] <= time_window:
                        count += 1
                        j += 1
                    
                    if count >= velocity_threshold:
                        # Add velocity pattern node
                        pattern_id = f"velocity_pattern_{hashlib.md5(f'{node_id}_{i}'.encode()).hexdigest()[:8]}"
                        self.fraud_graph.add_entity(Entity(
                            id=pattern_id,
                            entity_type="velocity_pattern",
                            attributes={
                                "transaction_count": count,
                                "time_window_hours": time_window.total_seconds() / 3600,
                                "start_time": transactions[i][1].isoformat(),
                                "risk_score": min(count / velocity_threshold, 1.0)
                            }
                        ))
                        
                        # Link card and transactions to pattern
                        self.fraud_graph.add_relationship(
                            node_id, pattern_id, "exhibits_pattern",
                            {"pattern_type": "high_velocity"}
                        )
                        
                        for k in range(i, min(j, len(transactions))):
                            self.fraud_graph.add_relationship(
                                transactions[k][0], pattern_id, "part_of_pattern",
                                {"pattern_type": "high_velocity"}
                            )
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the constructed graph"""
        stats = {
            "total_nodes": self.fraud_graph.graph.number_of_nodes(),
            "total_edges": self.fraud_graph.graph.number_of_edges(),
            "node_types": dict(self.fraud_graph.entity_counts),
            "edge_types": list(self.fraud_graph.edge_types),
            "density": nx.density(self.fraud_graph.graph),
            "is_connected": nx.is_weakly_connected(self.fraud_graph.graph),
            "num_components": nx.number_weakly_connected_components(self.fraud_graph.graph)
        }
        
        # Calculate degree statistics
        degrees = dict(self.fraud_graph.graph.degree())
        stats["avg_degree"] = np.mean(list(degrees.values()))
        stats["max_degree"] = max(degrees.values())
        stats["min_degree"] = min(degrees.values())
        
        return stats