from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import hashlib
import time

router = APIRouter()

class Record(BaseModel):
    record_id: str
    data_type: str  # "water" or "health"
    data: dict
    timestamp: float = Field(default_factory=lambda: time.time())
    stakeholder: str

class Block(BaseModel):
    index: int
    timestamp: float
    records: List[Record]
    previous_hash: str
    hash: str
    nonce: int

class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.current_records: List[Record] = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            records=[],
            previous_hash="0",
            hash="",
            nonce=0
        )
        genesis_block.hash = self.hash_block(genesis_block)
        self.chain.append(genesis_block)

    def add_record(self, record: Record):
        self.current_records.append(record)

    def mine_block(self):
        previous_block = self.chain[-1]
        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            records=self.current_records.copy(),
            previous_hash=previous_block.hash,
            hash="",
            nonce=0
        )
        block.hash = self.proof_of_work(block)
        self.chain.append(block)
        self.current_records.clear()
        return block

    def hash_block(self, block: Block) -> str:
        block_string = f"{block.index}{block.timestamp}{block.records}{block.previous_hash}{block.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def proof_of_work(self, block: Block) -> str:
        block.nonce = 0
        computed_hash = self.hash_block(block)
        while not computed_hash.startswith("0000"):
            block.nonce += 1
            computed_hash = self.hash_block(block)
        return computed_hash

    def get_chain(self) -> List[Block]:
        return self.chain

blockchain = Blockchain()

@router.post("/record", response_model=Record)
def add_record(record: Record):
    blockchain.add_record(record)
    return record

@router.post("/mine", response_model=Block)
def mine_block():
    if not blockchain.current_records:
        raise HTTPException(status_code=400, detail="No records to mine")
    block = blockchain.mine_block()
    return block

@router.get("/chain", response_model=List[Block])
def get_chain():
    return blockchain.get_chain()