#!/usr/bin/env python3
"""
CHIMERA Demo Suite Launcher
Central execution point for all demos and benchmarks

Author: Based on CHIMERA architecture by Francisco Angulo de Lafuente
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import json


class DemoLauncher:
    """Central launcher for all CHIMERA demos"""
    
    def __init__(self):
        self.demos = {
            '1': {
                'name': 'Performance Benchmark Comparison',
                'script': 'chimera_benchmark_comparison.py',
                'description': 'Compare CHIMERA vs PyTorch performance (43× speedup)',
                'duration': '~30 seconds',
                'category': 'benchmark'
            },
            '2': {
                'name': 'ARC-AGI Puzzle Solver',
                'script': 'chimera_arc_puzzle_demo.py',
                'description': 'Abstract reasoning with GPU textures (57.3% accuracy)',
                'duration': '~20 seconds',
                'category': 'reasoning'
            },
            '3': {
                'name': 'Real-Time Text Generation',
                'script': 'chimera_realtime_text_generation.py',
                'description': 'Parallel vs sequential generation (33× faster)',
                'duration': '~25 seconds',
                'category': 'nlp'
            },
            '4': {
                'name': 'Edge AI Deployment',
                'script': 'chimera_edge_ai_demo.py',
                'description': 'AI on resource-constrained devices (88.7% less memory)',
                'duration': '~15 seconds',
                'category': 'edge'
            },
            '5': {
                'name': 'Automated Benchmark Suite',
                'script': 'chimera_automated_benchmarks.py',
                'description': 'MLPerf, GLUE, ARC-AGI comprehensive testing',
                'duration': '~45 seconds',
                'category': 'benchmark'
            },
            '6': {
                'name': 'Integration Guide',
                'script': 'chimera_integration_guide.py',
                'description': 'How to integrate with CHIMERA repositories',
                'duration': '~10 seconds',
                'category': 'tutorial'
            }
        }
        
        self.results = {}
    
    def print_header(self):
        """Print fancy header"""
        print("\n" + "="*80)
        print("=" + " "*78 + "=")
        print("=" + "  CHIMERA DEMO SUITE LAUNCHER".center(78) + "=")
        print("=" + "  Revolutionary GPU-Native AI Architecture".center(78) + "=")
        print("=" + " "*78 + "=")
        print("="*80 + "\n")
    
    def print_menu(self):
        """Print demo selection menu"""
        print("=" * 80)
        print("AVAILABLE DEMOS")
        print("=" * 80 + "\n")
        
        # Group by category
        categories = {}
        for demo_id, demo_info in self.demos.items():
            cat = demo_info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((demo_id, demo_info))
        
        # Display by category
        cat_names = {
            'benchmark': '📊 Performance Benchmarks',
            'reasoning': '🧩 Abstract Reasoning',
            'nlp': '💬 Natural Language Processing',
            'edge': '📱 Edge AI Deployment',
            'tutorial': '📚 Tutorials & Integration'
        }
        
        for cat, demos in categories.items():
            print(f"\n{cat_names[cat]}")
            print("-" * 80)
            for demo_id, demo_info in demos:
                print(f"  [{demo_id}] {demo_info['name']}")
                print(f"      {demo_info['description']}")
                print(f"      Duration: {demo_info['duration']}\n")
        
        print("=" * 80)
        print("  [A] Run ALL demos (complete test suite)")
        print("  [Q] Quit")
        print("=" * 80 + "\n")
    
    def run_demo(self, demo_id: str) -> dict:
        """Run a specific demo"""
        if demo_id not in self.demos:
            return {'success': False, 'error': 'Invalid demo ID'}
        
        demo = self.demos[demo_id]
        
        print("\n" + "▼"*80)
        print(f"▼  RUNNING: {demo['name']}")
        print("▼"*80 + "\n")
        
        start_time = time.time()
        
        try:
            # Run the demo script
            result = subprocess.run(
                [sys.executable, demo['script']],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            elapsed = time.time() - start_time
            
            # Check for success
            success = result.returncode == 0
            
            return {
                'success': success,
                'duration': elapsed,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return {
                'success': False,
                'duration': elapsed,
                'error': 'Timeout (>120s)'
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                'success': False,
                'duration': elapsed,
                'error': str(e)
            }
    
    def run_all_demos(self):
        """Run all demos in sequence"""
        print("\n" + "="*80)
        print("=" + " "*78 + "=")
        print("=" + "  RUNNING COMPLETE DEMO SUITE".center(78) + "=")
        print("=" + " "*78 + "=")
        print("="*80 + "\n")
        
        total_start = time.time()
        
        for demo_id in sorted(self.demos.keys()):
            result = self.run_demo(demo_id)
            self.results[demo_id] = result
            
            if result['success']:
                print(f"\nSUCCESS: {self.demos[demo_id]['name']} - PASSED ({result['duration']:.1f}s)")
            else:
                print(f"\nFAILED: {self.demos[demo_id]['name']} - FAILED ({result['duration']:.1f}s)")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
            
            print("-" * 80)
        
        total_elapsed = time.time() - total_start
        
        # Generate summary
        self.generate_summary(total_elapsed)
    
    def generate_summary(self, total_time: float):
        """Generate execution summary"""
        print("\n" + "="*80)
        print("=" + " "*78 + "=")
        print("=" + "  DEMO SUITE SUMMARY".center(78) + "=")
        print("=" + " "*78 + "=")
        print("="*80 + "\n")
        
        passed = sum(1 for r in self.results.values() if r['success'])
        failed = len(self.results) - passed
        total_demos = len(self.results)
        
        print(f"Total Demos Run:    {total_demos}")
        print(f"Passed:             {passed} OK")
        print(f"Failed:             {failed} {'FAIL' if failed > 0 else 'OK'}")
        print(f"Total Time:         {total_time:.1f}s")
        print(f"Average Time/Demo:  {total_time/total_demos:.1f}s")
        
        print("\n" + "=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80 + "\n")
        
        for demo_id, result in self.results.items():
            demo = self.demos[demo_id]
            status = "OK PASS" if result['success'] else "FAIL"
            print(f"{status} | {demo['name']:<50} | {result['duration']:>6.1f}s")
        
        # Save results to JSON
        self.save_results()
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_suite_results_{timestamp}.json"
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_demos': len(self.results),
            'passed': sum(1 for r in self.results.values() if r['success']),
            'failed': sum(1 for r in self.results.values() if not r['success']),
            'results': {}
        }
        
        for demo_id, result in self.results.items():
            output['results'][demo_id] = {
                'name': self.demos[demo_id]['name'],
                'success': result['success'],
                'duration': result['duration'],
                'error': result.get('error', None)
            }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n📁 Results saved to: {filename}")
    
    def interactive_mode(self):
        """Interactive demo selection"""
        self.print_header()
        
        while True:
            self.print_menu()
            
            choice = input("Select demo (1-6, A for all, Q to quit): ").strip().upper()
            
            if choice == 'Q':
                print("\n👋 Thank you for using CHIMERA Demo Suite!")
                print("="*80)
                break
            elif choice == 'A':
                self.run_all_demos()
                input("\nPress Enter to return to menu...")
            elif choice in self.demos:
                result = self.run_demo(choice)
                if result['success']:
                    print(f"\nOK Demo completed successfully in {result['duration']:.1f}s")
                else:
                    print(f"\nFAIL Demo failed after {result['duration']:.1f}s")
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                input("\nPress Enter to return to menu...")
            else:
                print("\n⚠️  Invalid choice. Please try again.")
                time.sleep(1)
    
    def quick_start_guide(self):
        """Display quick start information"""
        print("\n" + "="*80)
        print("QUICK START GUIDE")
        print("="*80 + "\n")
        
        print("📚 Documentation Files:")
        print("   • README.md                          - Complete overview")
        print("   • CHIMERA_IMPLEMENTATION_ROADMAP.md  - Strategic roadmap")
        print()
        
        print("🚀 Run Individual Demos:")
        for demo_id, demo in self.demos.items():
            print(f"   • python {demo['script']}")
        print()
        
        print("⚡ Quick Commands:")
        print("   • python demo_launcher.py            - Interactive menu")
        print("   • python demo_launcher.py --all      - Run all demos")
        print("   • python demo_launcher.py --help     - Show this guide")
        print()
        
        print("🔗 CHIMERA Repositories:")
        print("   • github.com/Agnuxo1/Neuromorphic_GPU_Native_Intelligence_System")
        print("   • github.com/Agnuxo1/CHIMERA-Revolutionary-AI-Architecture")
        print("   • github.com/Agnuxo1/No-CUDA-No-Tensor-Cores-ALL-GPUs")
        print()
        
        print("📊 Key Results to Expect:")
        print("   • 43× speedup in matrix operations")
        print("   • 88.7% memory reduction vs PyTorch")
        print("   • 57.3% accuracy on ARC-AGI benchmark")
        print("   • Universal GPU compatibility")
        print("="*80 + "\n")


def main():
    """Main execution"""
    launcher = DemoLauncher()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h']:
            launcher.quick_start_guide()
        elif arg in ['--all', '-a']:
            launcher.print_header()
            launcher.run_all_demos()
        elif arg in ['--list', '-l']:
            launcher.print_header()
            launcher.print_menu()
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
    else:
        # Interactive mode
        launcher.interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
