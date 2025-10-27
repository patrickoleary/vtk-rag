#!/usr/bin/env python3
"""
Corpus Analysis Script

Analyzes chunked corpus files to provide statistics on:
- Chunk size distribution
- Token counts
- Overlap validation
- Content coverage

Optionally creates visualizations with matplotlib (install with: pip install matplotlib seaborn)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import statistics

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False


class ChunkAnalyzer:
    """Analyzer for chunked corpus files"""
    
    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return int(len(text) / self.chars_per_token)
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single chunked JSONL file"""
        print(f"\nAnalyzing: {file_path.name}")
        print("=" * 80)
        
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
        
        if not chunks:
            print("No chunks found!")
            return {}
        
        # Calculate statistics
        stats = self._calculate_statistics(chunks)
        self._print_statistics(stats)
        
        return stats
    
    def _calculate_statistics(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        stats = {
            'total_chunks': len(chunks),
            'source_types': Counter(),
            'chunk_sizes_chars': [],
            'chunk_sizes_tokens': [],
            'original_docs': set(),
            'chunks_per_doc': defaultdict(int),
            'metadata_fields': defaultdict(int),
            'has_overlap': 0,
            'categories': Counter(),
            'symbols_used': Counter()
        }
        
        for chunk in chunks:
            # Basic counts
            content = chunk.get('content', '')
            stats['chunk_sizes_chars'].append(len(content))
            stats['chunk_sizes_tokens'].append(self.estimate_tokens(content))
            
            # Source tracking
            source_type = chunk.get('source_type', 'unknown')
            stats['source_types'][source_type] += 1
            
            original_id = chunk.get('original_id', '')
            stats['original_docs'].add(original_id)
            stats['chunks_per_doc'][original_id] += 1
            
            # Overlap tracking
            if chunk.get('overlap_context'):
                stats['has_overlap'] += 1
            
            # Metadata analysis
            metadata = chunk.get('metadata', {})
            for key in metadata.keys():
                stats['metadata_fields'][key] += 1
            
            # Category tracking
            if 'category' in metadata:
                stats['categories'][metadata['category']] += 1
            
            # Symbol tracking
            if 'uses_symbols' in metadata:
                for symbol in metadata.get('uses_symbols', []):
                    stats['symbols_used'][symbol] += 1
        
        # Calculate distributions
        if stats['chunk_sizes_chars']:
            stats['char_distribution'] = {
                'min': min(stats['chunk_sizes_chars']),
                'max': max(stats['chunk_sizes_chars']),
                'mean': statistics.mean(stats['chunk_sizes_chars']),
                'median': statistics.median(stats['chunk_sizes_chars']),
                'stdev': statistics.stdev(stats['chunk_sizes_chars']) if len(stats['chunk_sizes_chars']) > 1 else 0
            }
        
        if stats['chunk_sizes_tokens']:
            stats['token_distribution'] = {
                'min': min(stats['chunk_sizes_tokens']),
                'max': max(stats['chunk_sizes_tokens']),
                'mean': statistics.mean(stats['chunk_sizes_tokens']),
                'median': statistics.median(stats['chunk_sizes_tokens']),
                'stdev': statistics.stdev(stats['chunk_sizes_tokens']) if len(stats['chunk_sizes_tokens']) > 1 else 0
            }
        
        stats['unique_docs'] = len(stats['original_docs'])
        stats['avg_chunks_per_doc'] = len(chunks) / stats['unique_docs'] if stats['unique_docs'] > 0 else 0
        
        return stats
    
    def _print_statistics(self, stats: Dict[str, Any]):
        """Print formatted statistics"""
        print(f"\nOVERALL STATISTICS")
        print("-" * 80)
        print(f"Total chunks:          {stats['total_chunks']:,}")
        print(f"Unique documents:      {stats['unique_docs']:,}")
        print(f"Avg chunks per doc:    {stats['avg_chunks_per_doc']:.2f}")
        print(f"Chunks with overlap:   {stats['has_overlap']:,} ({stats['has_overlap']/stats['total_chunks']*100:.1f}%)")
        
        print(f"\nSOURCE TYPE DISTRIBUTION")
        print("-" * 80)
        for source_type, count in stats['source_types'].most_common():
            pct = count / stats['total_chunks'] * 100
            print(f"{source_type:15} {count:6,} ({pct:5.1f}%)")
        
        if stats.get('char_distribution'):
            print(f"\nCHUNK SIZE (Characters)")
            print("-" * 80)
            dist = stats['char_distribution']
            print(f"Min:     {dist['min']:8,}")
            print(f"Max:     {dist['max']:8,}")
            print(f"Mean:    {dist['mean']:8,.0f}")
            print(f"Median:  {dist['median']:8,.0f}")
            print(f"StdDev:  {dist['stdev']:8,.0f}")
        
        if stats.get('token_distribution'):
            print(f"\nCHUNK SIZE (Estimated Tokens)")
            print("-" * 80)
            dist = stats['token_distribution']
            print(f"Min:     {dist['min']:8,}")
            print(f"Max:     {dist['max']:8,}")
            print(f"Mean:    {dist['mean']:8,.0f}")
            print(f"Median:  {dist['median']:8,.0f}")
            print(f"StdDev:  {dist['stdev']:8,.0f}")
        
        if stats['categories']:
            print(f"\nCATEGORY DISTRIBUTION (Examples)")
            print("-" * 80)
            for category, count in stats['categories'].most_common(10):
                category_str = str(category) if category is not None else "Uncategorized"
                print(f"{category_str:30} {count:5,}")
        
        if stats['symbols_used']:
            print(f"\nTOP VTK SYMBOLS USED")
            print("-" * 80)
            for symbol, count in stats['symbols_used'].most_common(20):
                print(f"{symbol:40} {count:5,}")
        
        # Chunk distribution
        chunk_counts = list(stats['chunks_per_doc'].values())
        if chunk_counts:
            print(f"\nCHUNKS PER DOCUMENT DISTRIBUTION")
            print("-" * 80)
            print(f"Min chunks:     {min(chunk_counts)}")
            print(f"Max chunks:     {max(chunk_counts)}")
            print(f"Mean chunks:    {statistics.mean(chunk_counts):.2f}")
            print(f"Median chunks:  {statistics.median(chunk_counts):.0f}")
            
            # Distribution bins
            bins = {1: 0, 2: 0, 3: 0, '4-5': 0, '6-10': 0, '11+': 0}
            for count in chunk_counts:
                if count == 1:
                    bins[1] += 1
                elif count == 2:
                    bins[2] += 1
                elif count == 3:
                    bins[3] += 1
                elif count <= 5:
                    bins['4-5'] += 1
                elif count <= 10:
                    bins['6-10'] += 1
                else:
                    bins['11+'] += 1
            
            print(f"\nDocument distribution by chunk count:")
            for bin_name, count in bins.items():
                print(f"  {str(bin_name):6} chunks: {count:5,} documents")
    
    def create_visualizations(self, stats: Dict[str, Any], output_dir: Path, prefix: str = "corpus"):
        """Create visualization plots from statistics"""
        if not VISUALIZATIONS_AVAILABLE:
            print("\n⚠️  Matplotlib not installed. Skipping visualizations.")
            print("   Install with: pip install matplotlib")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreating visualizations in {output_dir}/...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        
        # 1. Token distribution histogram
        if stats.get('chunk_sizes_tokens'):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(stats['chunk_sizes_tokens'], bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Token Count', fontsize=12)
            ax.set_ylabel('Number of Chunks', fontsize=12)
            ax.set_title('Chunk Size Distribution (Tokens)', fontsize=14, fontweight='bold')
            ax.axvline(statistics.mean(stats['chunk_sizes_tokens']), color='red', 
                      linestyle='--', linewidth=2, label=f"Mean: {statistics.mean(stats['chunk_sizes_tokens']):.0f}")
            ax.axvline(statistics.median(stats['chunk_sizes_tokens']), color='green', 
                      linestyle='--', linewidth=2, label=f"Median: {statistics.median(stats['chunk_sizes_tokens']):.0f}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'{prefix}_token_distribution.png', dpi=150)
            plt.close()
            print(f"  ✓ {prefix}_token_distribution.png")
        
        # 2. Source type bar chart - REMOVED (too simple, shown in text summary)
        
        # 3. Top categories bar chart
        if stats['categories']:
            fig, ax = plt.subplots(figsize=(12, 6))
            top_categories = stats['categories'].most_common(15)
            categories = [c if c else "Uncategorized" for c, _ in top_categories]
            counts = [count for _, count in top_categories]
            bars = ax.barh(range(len(categories)), counts, edgecolor='black', alpha=0.7)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels(categories)
            ax.set_xlabel('Number of Chunks', fontsize=12)
            ax.set_title('Top 15 Categories', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(count, i, f' {count:,}', va='center', fontsize=9)
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(output_dir / f'{prefix}_categories.png', dpi=150)
            plt.close()
            print(f"  ✓ {prefix}_categories.png")
        
        # 4. Top VTK symbols bar chart
        if stats['symbols_used']:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_symbols = stats['symbols_used'].most_common(20)
            symbols = [s for s, _ in top_symbols]
            counts = [c for _, c in top_symbols]
            bars = ax.barh(range(len(symbols)), counts, edgecolor='black', alpha=0.7)
            ax.set_yticks(range(len(symbols)))
            ax.set_yticklabels(symbols, fontsize=9)
            ax.set_xlabel('Usage Count', fontsize=12)
            ax.set_title('Top 20 VTK Symbols Used', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(count, i, f' {count:,}', va='center', fontsize=8)
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(output_dir / f'{prefix}_vtk_symbols.png', dpi=150)
            plt.close()
            print(f"  ✓ {prefix}_vtk_symbols.png")
        
        # 5. Box plot - REMOVED (redundant with histogram)
        
        print(f"\n✅ Visualizations saved to {output_dir}/")
    
    def analyze_all(self, directory: Path, visualize: bool = False, viz_output: Optional[Path] = None):
        """Analyze all chunked files in directory"""
        print("\n" + "=" * 80)
        print("VTK CORPUS ANALYSIS")
        print("=" * 80)
        
        all_stats = {}
        
        for file_path in sorted(directory.glob('*_chunks.jsonl')):
            stats = self.analyze_file(file_path)
            all_stats[file_path.stem] = stats
            
            # Create visualizations for this file if requested
            if visualize and viz_output:
                self.create_visualizations(stats, viz_output, prefix=file_path.stem)
        
        # Overall summary
        if all_stats:
            print("\n" + "=" * 80)
            print("COMBINED SUMMARY")
            print("=" * 80)
            
            total_chunks = sum(s.get('total_chunks', 0) for s in all_stats.values())
            total_docs = sum(s.get('unique_docs', 0) for s in all_stats.values())
            
            print(f"Total files analyzed:  {len(all_stats)}")
            print(f"Total chunks:          {total_chunks:,}")
            print(f"Total documents:       {total_docs:,}")
            print(f"Avg chunks per doc:    {total_chunks/total_docs:.2f}" if total_docs > 0 else "N/A")
            
            # Combined token distribution
            all_tokens = []
            for stats in all_stats.values():
                all_tokens.extend(stats.get('chunk_sizes_tokens', []))
            
            if all_tokens:
                print(f"\nCOMBINED TOKEN DISTRIBUTION")
                print("-" * 80)
                print(f"Min:     {min(all_tokens):8,}")
                print(f"Max:     {max(all_tokens):8,}")
                print(f"Mean:    {statistics.mean(all_tokens):8,.0f}")
                print(f"Median:  {statistics.median(all_tokens):8,.0f}")
                print(f"StdDev:  {statistics.stdev(all_tokens):8,.0f}" if len(all_tokens) > 1 else "N/A")
                
                # Percentiles
                sorted_tokens = sorted(all_tokens)
                p25 = sorted_tokens[len(sorted_tokens)//4]
                p75 = sorted_tokens[3*len(sorted_tokens)//4]
                p90 = sorted_tokens[9*len(sorted_tokens)//10]
                p95 = sorted_tokens[19*len(sorted_tokens)//20]
                
                print(f"\nPercentiles:")
                print(f"  25th: {p25:6,} tokens")
                print(f"  75th: {p75:6,} tokens")
                print(f"  90th: {p90:6,} tokens")
                print(f"  95th: {p95:6,} tokens")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze chunked VTK corpus files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all files
  python analyze_corpus.py
  
  # Analyze with visualizations
  python analyze_corpus.py --visualize
  
  # Analyze specific file with visualizations
  python analyze_corpus.py --file data/processed/code_chunks.jsonl --visualize
        """
    )
    parser.add_argument(
        '--directory',
        type=Path,
        default=Path('data/processed'),
        help='Directory containing chunked JSONL files'
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Analyze specific file instead of directory'
    )
    parser.add_argument(
        '--chars-per-token',
        type=float,
        default=4.0,
        help='Character per token ratio for estimation'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization charts (requires matplotlib)'
    )
    parser.add_argument(
        '--viz-output',
        type=Path,
        default=Path('data/visualizations'),
        help='Output directory for visualizations (default: data/visualizations/)'
    )
    
    args = parser.parse_args()
    
    # Check if visualization is requested but matplotlib not available
    if args.visualize and not VISUALIZATIONS_AVAILABLE:
        print("\n⚠️  Warning: --visualize flag set but matplotlib is not installed")
        print("   Install with: pip install matplotlib")
        print("   Continuing with text output only...\n")
    
    analyzer = ChunkAnalyzer(chars_per_token=args.chars_per_token)
    
    if args.file:
        stats = analyzer.analyze_file(args.file)
        if args.visualize and stats:
            analyzer.create_visualizations(stats, args.viz_output, 
                                          prefix=args.file.stem)
    else:
        analyzer.analyze_all(args.directory, visualize=args.visualize, 
                           viz_output=args.viz_output if args.visualize else None)


if __name__ == '__main__':
    main()
