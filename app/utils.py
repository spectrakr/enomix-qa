from typing import List, Dict, Any

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """참조 문서 정보를 포맷팅하는 공통 함수"""
    if not sources:
        return "\n\n📚 *참조 문서:* _없음_"
    
    source_texts = set()
    for source in sources:
        page = int(source['page']) + 1 if source['page'] != "N/A" else "N/A"
        source_texts.add(f"- 📄 `{source['source']}` p.{page}")
    return "\n\n📚 *참조 문서:*\n" + "\n".join(sorted(source_texts)) 