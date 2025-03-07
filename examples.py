from typing import Dict, Any
import numpy as np

def get_example_paradoxes() -> Dict[str, Dict[str, Any]]:
    """
    Return a dictionary of example paradoxes that users can select from.
    
    Each example includes a description, type, and the paradox itself.
    
    Returns:
        Dictionary of example paradoxes
    """
    examples = {
        "Liar Paradox": {
            "type": "Text",
            "paradox": "This statement is false.",
            "description": "A classic self-referential paradox where a statement contradicts itself."
        },
        "Russell's Paradox": {
            "type": "Text",
            "paradox": "Let R be the set of all sets that do not contain themselves. Does R contain itself?",
            "description": "A foundational paradox in set theory that challenged early formulations."
        },
        "Sorites Paradox": {
            "type": "Text",
            "paradox": "One grain of sand is not a heap. Adding one grain never makes a non-heap into a heap. Therefore, no amount of sand is a heap.",
            "description": "A paradox that explores vague predicates and the problem of drawing precise boundaries."
        },
        "Berry's Paradox": {
            "type": "Text",
            "paradox": "The smallest positive integer not definable in under sixty letters.",
            "description": "A paradox arising from the attempt to define a number that, by definition, cannot be defined."
        },
        "Fixed Point Equation": {
            "type": "Numerical",
            "paradox": "x = 1/x",
            "description": "A simple recursive equation where x equals its own reciprocal."
        },
        "Recursive Oscillation": {
            "type": "Numerical",
            "paradox": "x = -x",
            "description": "A recursive equation that seems to require a value to equal its negative."
        },
        "Convergent Recursion": {
            "type": "Numerical",
            "paradox": "x = (x + 2)/2",
            "description": "A recursive equation that converges to a fixed point when iterated."
        },
        "Chaotic Mapping": {
            "type": "Numerical",
            "paradox": "x = 3.9 * x * (1 - x)",
            "description": "A logistic map equation that exhibits chaotic behavior in certain ranges."
        },
        "Circular Identity Matrix": {
            "type": "Matrix",
            "paradox": np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ]),
            "description": "A matrix that represents a circular permutation, raising questions about eigenvalues and stability."
        },
        "Contradictory Constraints": {
            "type": "Matrix",
            "paradox": np.array([
                [1, 0.5],
                [0.5, -1]
            ]),
            "description": "A matrix with competing positive and negative influences, creating tension between elements."
        },
        "Unstable Feedback System": {
            "type": "Matrix",
            "paradox": np.array([
                [1.1, 0.3, 0.1],
                [0.2, 0.9, 0.4],
                [0.1, 0.2, 1.2]
            ]),
            "description": "A matrix representing a system with positive feedback loops that tends toward instability."
        },
        "Zeno's Dichotomy": {
            "type": "Text",
            "paradox": "Before reaching a destination, one must first reach the halfway point, but before that, one must reach the quarter way point, and so on infinitely. Therefore, motion is impossible.",
            "description": "One of Zeno's paradoxes that challenges the concept of infinite divisibility of space and time."
        },
        "Ship of Theseus": {
            "type": "Text", 
            "paradox": "If all parts of a ship are gradually replaced over time, is it still the same ship? At what point does it become a different ship?",
            "description": "A philosophical paradox about identity and the persistence of objects through change."
        },
        "Newcomb's Problem": {
            "type": "Text",
            "paradox": "An omniscient being presents two boxes. One contains $1,000. The other contains either $1,000,000 or nothing, based on the being's prediction of your choice. Do you choose both boxes or just the second?",
            "description": "A decision theory paradox that challenges notions of free will and determinism."
        },
        "Barber Paradox": {
            "type": "Text",
            "paradox": "In a village, the barber shaves all those, and only those, who do not shave themselves. Who shaves the barber?",
            "description": "Another form of Russell's paradox, presented in a more accessible context."
        }
    }
    
    return examples
