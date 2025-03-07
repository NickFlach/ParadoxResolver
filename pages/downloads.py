import streamlit as st
import base64

st.set_page_config(
    page_title="Downloads & Papers",
    page_icon="📥",
    layout="wide"
)

st.title("Downloads & Papers")

st.markdown("""
## Crypto_ParadoxOS Resources

Access our command-line interface (CLI) tools and technical papers to integrate Crypto_ParadoxOS
with your projects or learn more about the underlying technology.
""")

# Function to create a download link
def get_download_link(filename, text):
    with open(filename, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {text}</a>'
    return href

# Create CLI section
st.header("CLI Downloads")

st.markdown("""
### Command Line Interface (CLI)

Our CLI tool provides direct access to Crypto_ParadoxOS functionality from your terminal.
It supports all core features including paradox resolution, evolutionary engine operations,
and meta-resolver configuration.

#### System Requirements:
- **Windows**: Windows 10 or newer
- **Linux**: Ubuntu 20.04+ / CentOS 8+ / Debian 11+
- **Dependencies**: Python 3.8 or newer with numpy and matplotlib
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Windows CLI")
    st.markdown("""
    - Compatible with Windows 10/11
    - Requires Python 3.8+ with numpy and matplotlib
    - Simple command-line interface
    """)
    
    # Use the actual Windows CLI file
    st.markdown(get_download_link('paradox_cli_windows.exe', 'Windows CLI (64-bit)'), unsafe_allow_html=True)
    
    # Also offer the Python script version
    st.markdown(get_download_link('paradox_cli.py', 'Python Script (cross-platform)'), unsafe_allow_html=True)

with col2:
    st.subheader("Linux CLI")
    st.markdown("""
    - Compatible with major Linux distributions
    - Requires Python 3.8+ with numpy and matplotlib
    - Simple installation script included
    """)
    
    # Use the actual Linux CLI file
    st.markdown(get_download_link('paradox_cli_linux', 'Linux CLI (64-bit)'), unsafe_allow_html=True)
    
    # Also offer the shell script
    st.markdown(get_download_link('paradox_cli.sh', 'Bash Script (Linux/macOS)'), unsafe_allow_html=True)

# Usage instructions
st.subheader("CLI Usage")
st.code("""
# Basic paradox resolution
paradox_cli resolve --input "x = 1/x" --type numerical --iterations 20

# Evolutionary engine
paradox_cli evolve --generations 10 --population 20 --test-cases examples.json

# Meta-resolver with custom framework
paradox_cli meta-resolve --input "[[0.7, 0.3], [0.4, 0.6]]" --type matrix --framework custom_framework.json
""", language="bash")

# Technical Papers section
st.header("Technical Papers")

tab1, tab2 = st.tabs(["Whitepaper", "Yellow Paper"])

with tab1:
    st.subheader("Crypto_ParadoxOS: A Recursive Paradox-Resolution System")
    st.markdown("""
    Our whitepaper provides a comprehensive introduction to Crypto_ParadoxOS,
    its architecture, and applications. It covers the foundational concepts and
    use cases without diving deeply into technical implementation details.
    
    **Abstract**
    
    This paper introduces Crypto_ParadoxOS, an innovative computational system
    designed to resolve paradoxes through recursive transformation until an equilibrium
    state is reached. We present a framework that transcends traditional approaches to
    paradoxical problems by embracing both convergent recursive resolution and divergent
    informational expansion.
    
    The system demonstrates capabilities in resolving numerical, matrix-based, and
    textual paradoxes through a combination of fixed-point iteration, eigenvalue stabilization,
    and contradiction resolution techniques. Additionally, we introduce an evolutionary engine
    that generates novel transformation rules through genetic programming, enabling the system
    to discover creative solutions to previously intractable problems.
    
    We explore applications in decision-making under contradictory requirements, resource
    allocation with competing priorities, and AI ethics dilemmas, demonstrating how
    Crypto_ParadoxOS can address complex challenges across multiple domains.
    """)
    
    # Create a dummy whitepaper file
    with open('crypto_paradoxos_whitepaper.pdf', 'w') as f:
        f.write("This is a placeholder for the whitepaper PDF")
    
    st.markdown(get_download_link('crypto_paradoxos_whitepaper.pdf', 'Whitepaper (PDF)'), unsafe_allow_html=True)

with tab2:
    st.subheader("Crypto_ParadoxOS: Technical Implementation and Algorithms")
    st.markdown("""
    Our Yellow Paper provides a detailed technical specification of the
    Crypto_ParadoxOS system. It includes algorithm descriptions, implementation details,
    theoretical foundations, and formal proofs of key properties.
    
    **Abstract**
    
    This paper presents the formal specification and technical implementation details of
    Crypto_ParadoxOS, a recursive paradox-resolution system. We provide mathematical foundations
    for the transformation rules, convergence properties, and evolutionary mechanisms that
    drive the system.
    
    We introduce formal definitions for paradox states, transformation functions, and
    convergence criteria, along with proofs of termination and stability properties.
    The paper details the meta-resolver framework that orchestrates transitions between
    convergent and divergent processing phases, implementing a novel approach to the
    fundamental tension between recursive resolution and informational expansion.
    
    Additionally, we specify the evolutionary engine's genetic programming algorithms,
    including genome representation, mutation and crossover operators, and fitness evaluation
    mechanisms. We provide empirical results demonstrating the system's performance on
    benchmark paradoxes and analyze its computational complexity characteristics.
    
    Key algorithms are presented with pseudocode and implementation notes, enabling
    researchers and developers to understand, reproduce, and extend the system.
    """)
    
    # Create a dummy yellow paper file
    with open('crypto_paradoxos_yellowpaper.pdf', 'w') as f:
        f.write("This is a placeholder for the yellow paper PDF")
    
    st.markdown(get_download_link('crypto_paradoxos_yellowpaper.pdf', 'Yellow Paper (PDF)'), unsafe_allow_html=True)

# Contact information
st.header("Contact & Support")
st.markdown("""
For questions, support, or collaboration opportunities:

- **Email**: flaukowski@proton.me
- **GitHub**: [github.com/crypto-paradoxos](https://github.com/crypto-paradoxos)
- **Documentation**: [docs.crypto-paradoxos.io](https://docs.crypto-paradoxos.io)

We welcome feedback and contributions from researchers, developers, and users
interested in paradox resolution and evolutionary computation.
""")