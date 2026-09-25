import os
import sys
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*incompatible dtype.*"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def setup_environment():
    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_path = os.getcwd()

    if not os.path.exists(os.path.join(base_path, "orchestrator.py")):
        print(f"⚠️ Path Warning: orchestrator.py not found at {base_path}")
        print("❌ Error: Cannot locate project path.")

    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    os.chdir(base_path)
    return base_path


CURRENT_DIR = setup_environment()

try:
    from orchestrator import CentralOrchestrationAgent
    from nlp_processor import NLPProcessor
except ImportError as e:
    print(f"\n❌ Module Import Failed: {e}")
    print(f"   Current path: {os.getcwd()}")
    print("   Verify orchestrator.py exists in the same directory.")
    sys.exit(1)

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", module="sklearn")

    print("=" * 60)
    print("🤖 DiCerAgent v1.0")
    print("=" * 60)

    try:
        nlp = NLPProcessor(use_llm=True)
        agent = CentralOrchestrationAgent()
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        sys.exit(1)

    while True:
        try:
            raw_input = input(
                "\n💬 Enter Query, CSV Path, or Folder Path ('q' to quit): "
            ).strip()

            if raw_input.lower() in ["q", "exit", "quit"]:
                print("👋 Bye!")
                break

            if not raw_input:
                continue

            clean_path = raw_input.strip('"').strip("'")

            if os.path.isdir(clean_path):
                agent.run_directory(clean_path)
            elif os.path.isfile(clean_path) and clean_path.lower().endswith(".csv"):
                agent.run_batch(clean_path)
            else:
                agent_command = nlp.process_query(raw_input)
                if agent_command != raw_input:
                    print(f"   🧠 Parsed Command: '{agent_command}'")
                agent.run(agent_command)

        except KeyboardInterrupt:
            print("\n👋 Force Exit.")
            break
        except Exception as e:
            print(f"❌ Critical Runtime Error: {e}")
            import traceback

            traceback.print_exc()
