"""Debug Assistant service for AI-powered debugging assistance."""

import ast
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import selectinload

from ..models.debugging import (
    CodeContext, ErrorContext, ErrorAnalysis, DebuggingSolution, 
    DebuggingSession, DebuggingGuidanceStep, ErrorPattern, PotentialIssue,
    ErrorType, ComplexityLevel, SolutionStatus, DebuggingSessionStatus
)
from ..models.user import User, SkillLevel
from ..schemas.debugging import (
    CodeContextCreate, ErrorContextCreate, DebugAnalysisRequest,
    SolutionFeedback, DebuggingSessionCreate, GuidanceStepCreate,
    PotentialIssueQuery, DebugAnalysisResult
)
from ..logging_config import get_logger

logger = get_logger(__name__)


class CodeContextAnalyzer:
    """Analyzes code context for debugging assistance."""
    
    def __init__(self):
        self.supported_languages = {
            'python': self._analyze_python_context,
            'javascript': self._analyze_javascript_context,
            'typescript': self._analyze_typescript_context,
            'java': self._analyze_java_context,
            'csharp': self._analyze_csharp_context,
            'cpp': self._analyze_cpp_context,
            'go': self._analyze_go_context,
            'rust': self._analyze_rust_context
        }
    
    def analyze_code_structure(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code structure and extract relevant information."""
        try:
            analyzer = self.supported_languages.get(language.lower())
            if analyzer:
                return analyzer(code)
            else:
                return self._analyze_generic_context(code, language)
        except Exception as e:
            logger.error("Error analyzing code structure", error=str(e), language=language)
            return {"error": str(e), "language": language}
    
    def _analyze_python_context(self, code: str) -> Dict[str, Any]:
        """Analyze Python code context."""
        try:
            tree = ast.parse(code)
            
            context = {
                "language": "python",
                "functions": [],
                "classes": [],
                "imports": [],
                "variables": [],
                "decorators": [],
                "complexity_indicators": {}
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    context["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                        "is_async": isinstance(node, ast.AsyncFunctionDef)
                    })
                
                elif isinstance(node, ast.ClassDef):
                    context["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                        "decorators": [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                    })
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            context["imports"].append({
                                "module": alias.name,
                                "alias": alias.asname,
                                "line": node.lineno
                            })
                    else:  # ImportFrom
                        for alias in node.names:
                            context["imports"].append({
                                "module": node.module,
                                "name": alias.name,
                                "alias": alias.asname,
                                "line": node.lineno
                            })
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            context["variables"].append({
                                "name": target.id,
                                "line": node.lineno,
                                "type": "assignment"
                            })
            
            # Calculate complexity indicators
            context["complexity_indicators"] = {
                "total_lines": len(code.split('\n')),
                "function_count": len(context["functions"]),
                "class_count": len(context["classes"]),
                "import_count": len(context["imports"]),
                "cyclomatic_complexity": self._calculate_python_complexity(tree)
            }
            
            return context
            
        except SyntaxError as e:
            return {
                "language": "python",
                "syntax_error": {
                    "message": str(e),
                    "line": e.lineno,
                    "column": e.offset,
                    "text": e.text
                }
            }
        except Exception as e:
            logger.error("Error analyzing Python code", error=str(e))
            return {"error": str(e), "language": "python"}
    
    def _calculate_python_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity for Python code."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _analyze_javascript_context(self, code: str) -> Dict[str, Any]:
        """Analyze JavaScript code context using regex patterns."""
        context = {
            "language": "javascript",
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "complexity_indicators": {}
        }
        
        lines = code.split('\n')
        
        # Function patterns
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*\(',
            r'let\s+(\w+)\s*=\s*\(',
            r'var\s+(\w+)\s*=\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
            r'(\w+)\s*=>\s*'
        ]
        
        # Class patterns
        class_pattern = r'class\s+(\w+)'
        
        # Import patterns
        import_patterns = [
            r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'const\s+.*\s+=\s+require\([\'"]([^\'"]+)[\'"]\)',
            r'import\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
        ]
        
        # Variable patterns
        variable_patterns = [
            r'const\s+(\w+)',
            r'let\s+(\w+)',
            r'var\s+(\w+)'
        ]
        
        for i, line in enumerate(lines, 1):
            # Find functions
            for pattern in function_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    context["functions"].append({
                        "name": match,
                        "line": i,
                        "type": "function"
                    })
            
            # Find classes
            matches = re.findall(class_pattern, line)
            for match in matches:
                context["classes"].append({
                    "name": match,
                    "line": i
                })
            
            # Find imports
            for pattern in import_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    context["imports"].append({
                        "module": match,
                        "line": i
                    })
            
            # Find variables
            for pattern in variable_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    context["variables"].append({
                        "name": match,
                        "line": i,
                        "type": "variable"
                    })
        
        # Calculate complexity indicators
        context["complexity_indicators"] = {
            "total_lines": len(lines),
            "function_count": len(context["functions"]),
            "class_count": len(context["classes"]),
            "import_count": len(context["imports"]),
            "estimated_complexity": self._estimate_js_complexity(code)
        }
        
        return context
    
    def _estimate_js_complexity(self, code: str) -> int:
        """Estimate complexity for JavaScript code."""
        complexity = 1
        
        # Count control flow statements
        control_patterns = [
            r'\bif\s*\(',
            r'\belse\b',
            r'\bwhile\s*\(',
            r'\bfor\s*\(',
            r'\bswitch\s*\(',
            r'\bcase\s+',
            r'\btry\s*\{',
            r'\bcatch\s*\(',
            r'\?\s*.*\s*:'  # Ternary operator
        ]
        
        for pattern in control_patterns:
            complexity += len(re.findall(pattern, code))
        
        return complexity
    
    def _analyze_typescript_context(self, code: str) -> Dict[str, Any]:
        """Analyze TypeScript code context."""
        # TypeScript analysis is similar to JavaScript but with type information
        context = self._analyze_javascript_context(code)
        context["language"] = "typescript"
        
        # Add TypeScript-specific patterns
        lines = code.split('\n')
        
        # Interface patterns
        interface_pattern = r'interface\s+(\w+)'
        type_pattern = r'type\s+(\w+)'
        enum_pattern = r'enum\s+(\w+)'
        
        context["interfaces"] = []
        context["types"] = []
        context["enums"] = []
        
        for i, line in enumerate(lines, 1):
            # Find interfaces
            matches = re.findall(interface_pattern, line)
            for match in matches:
                context["interfaces"].append({
                    "name": match,
                    "line": i
                })
            
            # Find type definitions
            matches = re.findall(type_pattern, line)
            for match in matches:
                context["types"].append({
                    "name": match,
                    "line": i
                })
            
            # Find enums
            matches = re.findall(enum_pattern, line)
            for match in matches:
                context["enums"].append({
                    "name": match,
                    "line": i
                })
        
        return context
    
    def _analyze_java_context(self, code: str) -> Dict[str, Any]:
        """Analyze Java code context."""
        context = {
            "language": "java",
            "classes": [],
            "methods": [],
            "imports": [],
            "packages": [],
            "complexity_indicators": {}
        }
        
        lines = code.split('\n')
        
        # Java patterns
        class_pattern = r'(?:public|private|protected)?\s*class\s+(\w+)'
        method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\('
        import_pattern = r'import\s+([^;]+);'
        package_pattern = r'package\s+([^;]+);'
        
        for i, line in enumerate(lines, 1):
            # Find classes
            matches = re.findall(class_pattern, line)
            for match in matches:
                context["classes"].append({
                    "name": match,
                    "line": i
                })
            
            # Find methods
            matches = re.findall(method_pattern, line)
            for match in matches:
                context["methods"].append({
                    "name": match,
                    "line": i
                })
            
            # Find imports
            matches = re.findall(import_pattern, line)
            for match in matches:
                context["imports"].append({
                    "package": match.strip(),
                    "line": i
                })
            
            # Find package declaration
            matches = re.findall(package_pattern, line)
            for match in matches:
                context["packages"].append({
                    "name": match.strip(),
                    "line": i
                })
        
        context["complexity_indicators"] = {
            "total_lines": len(lines),
            "class_count": len(context["classes"]),
            "method_count": len(context["methods"]),
            "import_count": len(context["imports"])
        }
        
        return context
    
    def _analyze_csharp_context(self, code: str) -> Dict[str, Any]:
        """Analyze C# code context."""
        context = {
            "language": "csharp",
            "classes": [],
            "methods": [],
            "namespaces": [],
            "usings": [],
            "complexity_indicators": {}
        }
        
        lines = code.split('\n')
        
        # C# patterns
        class_pattern = r'(?:public|private|protected|internal)?\s*class\s+(\w+)'
        method_pattern = r'(?:public|private|protected|internal)?\s*(?:static)?\s*\w+\s+(\w+)\s*\('
        namespace_pattern = r'namespace\s+([^{]+)'
        using_pattern = r'using\s+([^;]+);'
        
        for i, line in enumerate(lines, 1):
            # Find classes
            matches = re.findall(class_pattern, line)
            for match in matches:
                context["classes"].append({
                    "name": match,
                    "line": i
                })
            
            # Find methods
            matches = re.findall(method_pattern, line)
            for match in matches:
                context["methods"].append({
                    "name": match,
                    "line": i
                })
            
            # Find namespaces
            matches = re.findall(namespace_pattern, line)
            for match in matches:
                context["namespaces"].append({
                    "name": match.strip(),
                    "line": i
                })
            
            # Find using statements
            matches = re.findall(using_pattern, line)
            for match in matches:
                context["usings"].append({
                    "namespace": match.strip(),
                    "line": i
                })
        
        context["complexity_indicators"] = {
            "total_lines": len(lines),
            "class_count": len(context["classes"]),
            "method_count": len(context["methods"]),
            "namespace_count": len(context["namespaces"])
        }
        
        return context
    
    def _analyze_cpp_context(self, code: str) -> Dict[str, Any]:
        """Analyze C++ code context."""
        context = {
            "language": "cpp",
            "functions": [],
            "classes": [],
            "includes": [],
            "namespaces": [],
            "complexity_indicators": {}
        }
        
        lines = code.split('\n')
        
        # C++ patterns
        function_pattern = r'(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:const)?\s*{'
        class_pattern = r'class\s+(\w+)'
        include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
        namespace_pattern = r'namespace\s+(\w+)'
        
        for i, line in enumerate(lines, 1):
            # Find functions
            matches = re.findall(function_pattern, line)
            for match in matches:
                context["functions"].append({
                    "name": match,
                    "line": i
                })
            
            # Find classes
            matches = re.findall(class_pattern, line)
            for match in matches:
                context["classes"].append({
                    "name": match,
                    "line": i
                })
            
            # Find includes
            matches = re.findall(include_pattern, line)
            for match in matches:
                context["includes"].append({
                    "header": match,
                    "line": i
                })
            
            # Find namespaces
            matches = re.findall(namespace_pattern, line)
            for match in matches:
                context["namespaces"].append({
                    "name": match,
                    "line": i
                })
        
        context["complexity_indicators"] = {
            "total_lines": len(lines),
            "function_count": len(context["functions"]),
            "class_count": len(context["classes"]),
            "include_count": len(context["includes"])
        }
        
        return context
    
    def _analyze_go_context(self, code: str) -> Dict[str, Any]:
        """Analyze Go code context."""
        context = {
            "language": "go",
            "functions": [],
            "types": [],
            "imports": [],
            "packages": [],
            "complexity_indicators": {}
        }
        
        lines = code.split('\n')
        
        # Go patterns
        function_pattern = r'func\s+(?:\([^)]*\)\s+)?(\w+)\s*\('
        type_pattern = r'type\s+(\w+)\s+'
        import_pattern = r'import\s+"([^"]+)"'
        package_pattern = r'package\s+(\w+)'
        
        for i, line in enumerate(lines, 1):
            # Find functions
            matches = re.findall(function_pattern, line)
            for match in matches:
                context["functions"].append({
                    "name": match,
                    "line": i
                })
            
            # Find types
            matches = re.findall(type_pattern, line)
            for match in matches:
                context["types"].append({
                    "name": match,
                    "line": i
                })
            
            # Find imports
            matches = re.findall(import_pattern, line)
            for match in matches:
                context["imports"].append({
                    "package": match,
                    "line": i
                })
            
            # Find package declaration
            matches = re.findall(package_pattern, line)
            for match in matches:
                context["packages"].append({
                    "name": match,
                    "line": i
                })
        
        context["complexity_indicators"] = {
            "total_lines": len(lines),
            "function_count": len(context["functions"]),
            "type_count": len(context["types"]),
            "import_count": len(context["imports"])
        }
        
        return context
    
    def _analyze_rust_context(self, code: str) -> Dict[str, Any]:
        """Analyze Rust code context."""
        context = {
            "language": "rust",
            "functions": [],
            "structs": [],
            "enums": [],
            "traits": [],
            "imports": [],
            "complexity_indicators": {}
        }
        
        lines = code.split('\n')
        
        # Rust patterns
        function_pattern = r'fn\s+(\w+)\s*\('
        struct_pattern = r'struct\s+(\w+)'
        enum_pattern = r'enum\s+(\w+)'
        trait_pattern = r'trait\s+(\w+)'
        use_pattern = r'use\s+([^;]+);'
        
        for i, line in enumerate(lines, 1):
            # Find functions
            matches = re.findall(function_pattern, line)
            for match in matches:
                context["functions"].append({
                    "name": match,
                    "line": i
                })
            
            # Find structs
            matches = re.findall(struct_pattern, line)
            for match in matches:
                context["structs"].append({
                    "name": match,
                    "line": i
                })
            
            # Find enums
            matches = re.findall(enum_pattern, line)
            for match in matches:
                context["enums"].append({
                    "name": match,
                    "line": i
                })
            
            # Find traits
            matches = re.findall(trait_pattern, line)
            for match in matches:
                context["traits"].append({
                    "name": match,
                    "line": i
                })
            
            # Find use statements
            matches = re.findall(use_pattern, line)
            for match in matches:
                context["imports"].append({
                    "module": match.strip(),
                    "line": i
                })
        
        context["complexity_indicators"] = {
            "total_lines": len(lines),
            "function_count": len(context["functions"]),
            "struct_count": len(context["structs"]),
            "enum_count": len(context["enums"]),
            "trait_count": len(context["traits"])
        }
        
        return context
    
    def _analyze_generic_context(self, code: str, language: str) -> Dict[str, Any]:
        """Generic code analysis for unsupported languages."""
        lines = code.split('\n')
        
        return {
            "language": language,
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "comment_lines": len([line for line in lines if line.strip().startswith(('#', '//', '/*', '*'))]),
            "estimated_complexity": max(1, len(lines) // 10),  # Rough estimate
            "analysis_type": "generic"
        }


class DebugAssistant:
    """Main Debug Assistant service for AI-powered debugging assistance."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.code_analyzer = CodeContextAnalyzer()
        
        # Error type classification patterns
        self.error_patterns = {
            ErrorType.SYNTAX_ERROR: [
                r'SyntaxError', r'invalid syntax', r'unexpected token', r'missing semicolon',
                r'unterminated string', r'unmatched parentheses'
            ],
            ErrorType.RUNTIME_ERROR: [
                r'RuntimeError', r'execution failed', r'runtime exception'
            ],
            ErrorType.TYPE_ERROR: [
                r'TypeError', r'type mismatch', r'cannot convert', r'wrong type'
            ],
            ErrorType.IMPORT_ERROR: [
                r'ImportError', r'ModuleNotFoundError', r'cannot import', r'module not found'
            ],
            ErrorType.ATTRIBUTE_ERROR: [
                r'AttributeError', r'has no attribute', r'undefined property'
            ],
            ErrorType.INDEX_ERROR: [
                r'IndexError', r'list index out of range', r'array index out of bounds'
            ],
            ErrorType.KEY_ERROR: [
                r'KeyError', r'key not found', r'undefined key'
            ],
            ErrorType.VALUE_ERROR: [
                r'ValueError', r'invalid value', r'value out of range'
            ],
            ErrorType.NETWORK_ERROR: [
                r'NetworkError', r'connection failed', r'timeout', r'network unreachable'
            ],
            ErrorType.DATABASE_ERROR: [
                r'DatabaseError', r'connection refused', r'query failed', r'table not found'
            ]
        }
    
    async def create_code_context(self, user_id: UUID, context_data: CodeContextCreate) -> CodeContext:
        """Create and store code context."""
        try:
            # Analyze code structure if content is provided
            code_analysis = {}
            if context_data.current_file_content:
                code_analysis = self.code_analyzer.analyze_code_structure(
                    context_data.current_file_content,
                    context_data.primary_language
                )
            
            # Generate context hash for deduplication
            context_hash = self._generate_context_hash(context_data)
            
            # Check if similar context already exists
            existing_context = await self._find_similar_context(user_id, context_hash)
            if existing_context:
                # Update existing context
                existing_context.current_file_content = context_data.current_file_content
                existing_context.cursor_position = context_data.cursor_position
                existing_context.selection_range = context_data.selection_range
                existing_context.updated_at = datetime.utcnow()
                await self.db.commit()
                return existing_context
            
            # Create new context
            code_context = CodeContext(
                user_id=user_id,
                workspace_root_path=context_data.workspace_root_path,
                project_type=context_data.project_type,
                primary_language=context_data.primary_language,
                framework=context_data.framework,
                dependencies=context_data.dependencies,
                current_file_path=context_data.current_file_path,
                current_file_content=context_data.current_file_content,
                cursor_position=context_data.cursor_position,
                selection_range=context_data.selection_range,
                project_structure={**context_data.project_structure, "code_analysis": code_analysis},
                git_branch=context_data.git_branch,
                recent_commits=context_data.recent_commits,
                changed_files=context_data.changed_files,
                ide_name=context_data.ide_name,
                ide_plugins=context_data.ide_plugins,
                environment_settings=context_data.environment_settings,
                context_hash=context_hash
            )
            
            self.db.add(code_context)
            await self.db.commit()
            await self.db.refresh(code_context)
            
            logger.info(
                "Code context created",
                user_id=str(user_id),
                context_id=str(code_context.id),
                language=context_data.primary_language
            )
            
            return code_context
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating code context", error=str(e), user_id=str(user_id))
            raise
    
    async def create_error_context(self, user_id: UUID, error_data: ErrorContextCreate, code_context_id: Optional[UUID] = None) -> ErrorContext:
        """Create and store error context."""
        try:
            # Classify error type
            classified_type = self._classify_error_type(error_data.error_message, error_data.stack_trace)
            if classified_type:
                error_data.error_type = classified_type
            
            # Generate error hash for deduplication
            error_hash = self._generate_error_hash(error_data)
            
            # Check if similar error already exists
            existing_error = await self._find_similar_error(user_id, error_hash)
            if existing_error:
                # Update frequency and last occurrence
                existing_error.frequency_count += 1
                existing_error.last_occurrence = datetime.utcnow()
                await self.db.commit()
                return existing_error
            
            # Create new error context
            error_context = ErrorContext(
                user_id=user_id,
                code_context_id=code_context_id,
                error_type=error_data.error_type,
                error_message=error_data.error_message,
                stack_trace=error_data.stack_trace,
                file_path=error_data.file_path,
                line_number=error_data.line_number,
                column_number=error_data.column_number,
                surrounding_code=error_data.surrounding_code,
                error_context_data=error_data.error_context_data,
                error_hash=error_hash
            )
            
            self.db.add(error_context)
            await self.db.commit()
            await self.db.refresh(error_context)
            
            logger.info(
                "Error context created",
                user_id=str(user_id),
                error_id=str(error_context.id),
                error_type=error_data.error_type
            )
            
            return error_context
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating error context", error=str(e), user_id=str(user_id))
            raise
    
    def _generate_context_hash(self, context_data: CodeContextCreate) -> str:
        """Generate hash for code context deduplication."""
        hash_data = {
            "workspace_root_path": context_data.workspace_root_path,
            "project_type": context_data.project_type,
            "primary_language": context_data.primary_language,
            "framework": context_data.framework,
            "current_file_path": context_data.current_file_path
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def _generate_error_hash(self, error_data: ErrorContextCreate) -> str:
        """Generate hash for error deduplication."""
        hash_data = {
            "error_type": error_data.error_type.value,
            "error_message": error_data.error_message[:200],  # First 200 chars
            "file_path": error_data.file_path,
            "line_number": error_data.line_number
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    async def _find_similar_context(self, user_id: UUID, context_hash: str) -> Optional[CodeContext]:
        """Find similar existing code context."""
        stmt = select(CodeContext).where(
            and_(
                CodeContext.user_id == user_id,
                CodeContext.context_hash == context_hash,
                CodeContext.is_active == True
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _find_similar_error(self, user_id: UUID, error_hash: str) -> Optional[ErrorContext]:
        """Find similar existing error context."""
        # Look for errors in the last 24 hours to avoid very old duplicates
        recent_threshold = datetime.utcnow() - timedelta(hours=24)
        
        stmt = select(ErrorContext).where(
            and_(
                ErrorContext.user_id == user_id,
                ErrorContext.error_hash == error_hash,
                ErrorContext.last_occurrence >= recent_threshold
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    def _classify_error_type(self, error_message: str, stack_trace: Optional[str] = None) -> Optional[ErrorType]:
        """Classify error type based on message and stack trace."""
        text_to_analyze = f"{error_message} {stack_trace or ''}".lower()
        
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern.lower(), text_to_analyze):
                    return error_type
        
        return None  # Return None if no pattern matches, will use provided type
    
    async def get_user_code_contexts(self, user_id: UUID, limit: int = 10) -> List[CodeContext]:
        """Get recent code contexts for a user."""
        stmt = select(CodeContext).where(
            and_(
                CodeContext.user_id == user_id,
                CodeContext.is_active == True
            )
        ).order_by(desc(CodeContext.updated_at)).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_user_error_contexts(self, user_id: UUID, limit: int = 20) -> List[ErrorContext]:
        """Get recent error contexts for a user."""
        stmt = select(ErrorContext).where(
            ErrorContext.user_id == user_id
        ).order_by(desc(ErrorContext.last_occurrence)).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def predict_potential_issues(self, code_context: CodeContext) -> List[Dict[str, Any]]:
        """Predict potential issues based on code context analysis."""
        try:
            potential_issues = []
            
            # Analyze project structure for potential issues
            project_structure = code_context.project_structure
            code_analysis = project_structure.get("code_analysis", {})
            
            if code_analysis.get("language") == "python":
                potential_issues.extend(self._predict_python_issues(code_analysis, code_context))
            elif code_analysis.get("language") in ["javascript", "typescript"]:
                potential_issues.extend(self._predict_js_issues(code_analysis, code_context))
            
            # Generic predictions based on project characteristics
            potential_issues.extend(self._predict_generic_issues(code_context))
            
            return potential_issues
            
        except Exception as e:
            logger.error("Error predicting potential issues", error=str(e))
            return []
    
    def _predict_python_issues(self, code_analysis: Dict[str, Any], code_context: CodeContext) -> List[Dict[str, Any]]:
        """Predict Python-specific potential issues."""
        issues = []
        
        # Check for syntax errors
        if "syntax_error" in code_analysis:
            issues.append({
                "type": ErrorType.SYNTAX_ERROR,
                "description": f"Syntax error detected: {code_analysis['syntax_error']['message']}",
                "likelihood": 0.9,
                "severity": 0.8,
                "location": {
                    "line": code_analysis['syntax_error'].get('line'),
                    "column": code_analysis['syntax_error'].get('column')
                }
            })
        
        # Check complexity
        complexity = code_analysis.get("complexity_indicators", {}).get("cyclomatic_complexity", 0)
        if complexity > 10:
            issues.append({
                "type": ErrorType.LOGIC_ERROR,
                "description": f"High cyclomatic complexity ({complexity}) may lead to logic errors",
                "likelihood": 0.6,
                "severity": 0.5,
                "prevention_tips": [
                    "Consider breaking down complex functions",
                    "Add unit tests for complex logic",
                    "Use early returns to reduce nesting"
                ]
            })
        
        # Check for missing imports
        functions = code_analysis.get("functions", [])
        imports = code_analysis.get("imports", [])
        imported_modules = {imp.get("module", "") for imp in imports}
        
        # Common Python modules that might be missing
        common_modules = ["os", "sys", "json", "datetime", "re"]
        for func in functions:
            func_name = func.get("name", "")
            if any(module in func_name.lower() for module in common_modules):
                for module in common_modules:
                    if module in func_name.lower() and module not in imported_modules:
                        issues.append({
                            "type": ErrorType.IMPORT_ERROR,
                            "description": f"Function '{func_name}' might need '{module}' import",
                            "likelihood": 0.4,
                            "severity": 0.6,
                            "prevention_tips": [f"Add 'import {module}' at the top of the file"]
                        })
        
        return issues
    
    def _predict_js_issues(self, code_analysis: Dict[str, Any], code_context: CodeContext) -> List[Dict[str, Any]]:
        """Predict JavaScript/TypeScript-specific potential issues."""
        issues = []
        
        # Check for potential undefined variables
        functions = code_analysis.get("functions", [])
        variables = code_analysis.get("variables", [])
        
        if len(functions) > len(variables) * 2:
            issues.append({
                "type": ErrorType.RUNTIME_ERROR,
                "description": "High function-to-variable ratio might indicate undefined variable usage",
                "likelihood": 0.3,
                "severity": 0.4,
                "prevention_tips": [
                    "Use 'const' and 'let' instead of 'var'",
                    "Enable strict mode",
                    "Use TypeScript for better type checking"
                ]
            })
        
        # Check for missing dependencies
        imports = code_analysis.get("imports", [])
        dependencies = code_context.dependencies or []
        
        imported_packages = {imp.get("module", "").split("/")[0] for imp in imports}
        declared_deps = {dep.get("name", "") for dep in dependencies}
        
        for package in imported_packages:
            if package and package not in declared_deps and not package.startswith("."):
                issues.append({
                    "type": ErrorType.IMPORT_ERROR,
                    "description": f"Package '{package}' is imported but not in dependencies",
                    "likelihood": 0.7,
                    "severity": 0.8,
                    "prevention_tips": [f"Add '{package}' to package.json dependencies"]
                })
        
        return issues
    
    def _predict_generic_issues(self, code_context: CodeContext) -> List[Dict[str, Any]]:
        """Predict generic potential issues."""
        issues = []
        
        # Check for large files
        if code_context.current_file_content:
            line_count = len(code_context.current_file_content.split('\n'))
            if line_count > 500:
                issues.append({
                    "type": ErrorType.PERFORMANCE_ERROR,
                    "description": f"Large file ({line_count} lines) may have performance or maintainability issues",
                    "likelihood": 0.4,
                    "severity": 0.3,
                    "prevention_tips": [
                        "Consider breaking the file into smaller modules",
                        "Extract reusable functions",
                        "Use proper code organization patterns"
                    ]
                })
        
        # Check for outdated dependencies
        dependencies = code_context.dependencies or []
        if len(dependencies) > 20:
            issues.append({
                "type": ErrorType.DEPENDENCY_ERROR,
                "description": f"Large number of dependencies ({len(dependencies)}) increases risk of conflicts",
                "likelihood": 0.3,
                "severity": 0.4,
                "prevention_tips": [
                    "Regularly audit and update dependencies",
                    "Remove unused dependencies",
                    "Use dependency scanning tools"
                ]
            })
        
        return issues