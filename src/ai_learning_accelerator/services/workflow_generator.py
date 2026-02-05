"""Automation workflow generator for creating executable scripts from patterns."""

import json
import logging
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_async_db
from ..models.automation import (
    ActionPattern, AutomationOpportunity, AutomationScript, AutomationExecution,
    AutomationStatus, ExecutionStatus, PatternType, ActionType
)
from ..models.user import User

logger = logging.getLogger(__name__)


class WorkflowGenerator:
    """Generates automation workflows from detected patterns."""
    
    def __init__(self):
        self.script_templates = {
            "bash": self._get_bash_template(),
            "python": self._get_python_template(),
            "powershell": self._get_powershell_template(),
            "batch": self._get_batch_template()
        }
    
    async def generate_workflow_from_opportunity(
        self,
        db: AsyncSession,
        opportunity_id: UUID,
        user_id: UUID
    ) -> Optional[AutomationScript]:
        """Generate an automation script from an opportunity."""
        try:
            # Get the opportunity and its pattern
            opportunity_query = select(AutomationOpportunity).where(
                and_(
                    AutomationOpportunity.id == opportunity_id,
                    AutomationOpportunity.user_id == user_id
                )
            )
            opportunity_result = await db.execute(opportunity_query)
            opportunity = opportunity_result.scalar_one_or_none()
            
            if not opportunity:
                logger.warning(f"Opportunity {opportunity_id} not found for user {user_id}")
                return None
            
            # Get the related pattern
            pattern_query = select(ActionPattern).where(ActionPattern.id == opportunity.pattern_id)
            pattern_result = await db.execute(pattern_query)
            pattern = pattern_result.scalar_one_or_none()
            
            if not pattern:
                logger.warning(f"Pattern {opportunity.pattern_id} not found")
                return None
            
            # Generate script based on pattern type
            script_content = await self._generate_script_content(pattern, opportunity)
            if not script_content:
                logger.warning(f"Could not generate script content for opportunity {opportunity_id}")
                return None
            
            # Determine script type based on pattern and environment
            script_type = self._determine_script_type(pattern)
            
            # Create automation script
            automation_script = AutomationScript(
                user_id=user_id,
                opportunity_id=opportunity_id,
                script_name=f"automation_{opportunity.title.lower().replace(' ', '_')}",
                script_description=f"Automated workflow for: {opportunity.description}",
                script_type=script_type,
                script_content=script_content,
                configuration=self._generate_script_configuration(pattern, opportunity),
                environment_requirements=self._get_environment_requirements(pattern),
                version="1.0.0",
                created_by="workflow_generator",
                is_enabled=False,  # Require user approval
                auto_execute=False,
                requires_confirmation=True,
                execution_count=0,
                success_count=0,
                total_time_saved_minutes=0,
                status=AutomationStatus.DETECTED
            )
            
            db.add(automation_script)
            await db.commit()
            await db.refresh(automation_script)
            
            logger.info(f"Generated automation script {automation_script.id} for opportunity {opportunity_id}")
            return automation_script
            
        except Exception as e:
            logger.error(f"Error generating workflow from opportunity {opportunity_id}: {e}")
            await db.rollback()
            return None
    
    async def execute_automation_script(
        self,
        db: AsyncSession,
        script_id: UUID,
        user_id: UUID,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Optional[AutomationExecution]:
        """Execute an automation script."""
        try:
            # Get the script
            script_query = select(AutomationScript).where(
                and_(
                    AutomationScript.id == script_id,
                    AutomationScript.user_id == user_id,
                    AutomationScript.is_enabled == True
                )
            )
            script_result = await db.execute(script_query)
            script = script_result.scalar_one_or_none()
            
            if not script:
                logger.warning(f"Script {script_id} not found or not enabled for user {user_id}")
                return None
            
            # Create execution record
            execution = AutomationExecution(
                script_id=script_id,
                user_id=user_id,
                execution_trigger="manual",
                execution_context=execution_context or {},
                started_at=datetime.utcnow(),
                status=ExecutionStatus.RUNNING
            )
            
            db.add(execution)
            await db.commit()
            await db.refresh(execution)
            
            # Execute the script
            execution_result = await self._execute_script_safely(script, execution_context or {})
            
            # Update execution record
            execution.completed_at = datetime.utcnow()
            execution.duration_ms = int((execution.completed_at - execution.started_at).total_seconds() * 1000)
            execution.status = ExecutionStatus.COMPLETED if execution_result["success"] else ExecutionStatus.FAILED
            execution.exit_code = execution_result.get("exit_code", 0)
            execution.output = execution_result.get("output", "")
            execution.error_output = execution_result.get("error", "")
            execution.files_processed = execution_result.get("files_processed", 0)
            
            # Update script statistics
            script.execution_count += 1
            if execution_result["success"]:
                script.success_count += 1
                # Estimate time saved (could be improved with better metrics)
                estimated_time_saved = max(1, script.opportunity.time_saving_potential // 60)  # Convert to minutes
                script.total_time_saved_minutes += estimated_time_saved
                execution.time_saved_minutes = estimated_time_saved
            
            script.last_executed_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(execution)
            
            logger.info(f"Executed automation script {script_id} with status {execution.status}")
            return execution
            
        except Exception as e:
            logger.error(f"Error executing automation script {script_id}: {e}")
            await db.rollback()
            return None
    
    async def get_execution_history(
        self,
        db: AsyncSession,
        user_id: UUID,
        script_id: Optional[UUID] = None,
        limit: int = 50
    ) -> List[AutomationExecution]:
        """Get execution history for a user."""
        try:
            query = select(AutomationExecution).where(AutomationExecution.user_id == user_id)
            
            if script_id:
                query = query.where(AutomationExecution.script_id == script_id)
            
            query = query.order_by(desc(AutomationExecution.started_at)).limit(limit)
            
            result = await db.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Error getting execution history for user {user_id}: {e}")
            return []
    
    async def get_automation_metrics(
        self,
        db: AsyncSession,
        user_id: UUID,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get automation metrics for a user."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get executions in the time period
            executions_query = select(AutomationExecution).where(
                and_(
                    AutomationExecution.user_id == user_id,
                    AutomationExecution.started_at >= start_date
                )
            )
            executions_result = await db.execute(executions_query)
            executions = executions_result.scalars().all()
            
            # Calculate metrics
            total_executions = len(executions)
            successful_executions = len([e for e in executions if e.status == ExecutionStatus.COMPLETED])
            failed_executions = len([e for e in executions if e.status == ExecutionStatus.FAILED])
            total_time_saved = sum([e.time_saved_minutes or 0 for e in executions])
            
            # Get active scripts
            scripts_query = select(AutomationScript).where(
                and_(
                    AutomationScript.user_id == user_id,
                    AutomationScript.is_enabled == True
                )
            )
            scripts_result = await db.execute(scripts_query)
            active_scripts = len(scripts_result.scalars().all())
            
            return {
                "period_days": days,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
                "total_time_saved_minutes": total_time_saved,
                "average_time_saved_per_execution": total_time_saved / total_executions if total_executions > 0 else 0,
                "active_scripts": active_scripts,
                "executions_per_day": total_executions / days if days > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting automation metrics for user {user_id}: {e}")
            return {}
    
    async def _generate_script_content(
        self,
        pattern: ActionPattern,
        opportunity: AutomationOpportunity
    ) -> Optional[str]:
        """Generate script content based on pattern and opportunity."""
        try:
            script_type = self._determine_script_type(pattern)
            template = self.script_templates.get(script_type)
            
            if not template:
                logger.warning(f"No template found for script type {script_type}")
                return None
            
            # Extract actions from pattern
            actions = pattern.action_sequence
            if not actions:
                logger.warning(f"No actions found in pattern {pattern.id}")
                return None
            
            # Generate script commands based on actions
            commands = []
            for action in actions:
                command = self._generate_command_from_action(action, script_type)
                if command:
                    commands.append(command)
            
            if not commands:
                logger.warning(f"No commands generated from pattern {pattern.id}")
                return None
            
            # Fill template with generated commands
            script_content = template.format(
                script_name=opportunity.title,
                description=opportunity.description,
                commands="\n".join(commands),
                timestamp=datetime.utcnow().isoformat()
            )
            
            return script_content
            
        except Exception as e:
            logger.error(f"Error generating script content: {e}")
            return None
    
    def _determine_script_type(self, pattern: ActionPattern) -> str:
        """Determine the best script type for a pattern."""
        # Analyze the actions to determine the best script type
        actions = pattern.action_sequence
        
        # Count different types of actions
        file_operations = sum(1 for action in actions if action.get("type") == ActionType.FILE_OPERATION.value)
        command_executions = sum(1 for action in actions if action.get("type") == ActionType.COMMAND_EXECUTION.value)
        build_operations = sum(1 for action in actions if action.get("type") == ActionType.BUILD_OPERATION.value)
        
        # Determine platform (simplified logic)
        has_windows_commands = any("cmd" in str(action).lower() or "powershell" in str(action).lower() for action in actions)
        has_unix_commands = any("bash" in str(action).lower() or "sh" in str(action).lower() for action in actions)
        
        # Decision logic
        if has_windows_commands:
            return "powershell"
        elif has_unix_commands:
            return "bash"
        elif build_operations > 0 or command_executions > file_operations:
            return "bash"  # Default to bash for command-heavy workflows
        else:
            return "python"  # Default to python for complex logic
    
    def _generate_command_from_action(self, action: Dict[str, Any], script_type: str) -> Optional[str]:
        """Generate a command from an action based on script type."""
        try:
            action_type = action.get("type")
            action_data = action.get("data", {})
            
            if action_type == ActionType.FILE_OPERATION.value:
                return self._generate_file_operation_command(action_data, script_type)
            elif action_type == ActionType.COMMAND_EXECUTION.value:
                return self._generate_command_execution(action_data, script_type)
            elif action_type == ActionType.BUILD_OPERATION.value:
                return self._generate_build_operation_command(action_data, script_type)
            elif action_type == ActionType.TEST_EXECUTION.value:
                return self._generate_test_execution_command(action_data, script_type)
            else:
                # Generic command
                command = action_data.get("command", "")
                if command:
                    return f"# {action_type}\n{command}"
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating command from action: {e}")
            return None
    
    def _generate_file_operation_command(self, action_data: Dict[str, Any], script_type: str) -> str:
        """Generate file operation command."""
        operation = action_data.get("operation", "")
        file_path = action_data.get("file_path", "")
        
        if script_type == "bash":
            if operation == "create":
                return f"touch '{file_path}'"
            elif operation == "delete":
                return f"rm -f '{file_path}'"
            elif operation == "copy":
                dest = action_data.get("destination", "")
                return f"cp '{file_path}' '{dest}'"
        elif script_type == "powershell":
            if operation == "create":
                return f"New-Item -Path '{file_path}' -ItemType File -Force"
            elif operation == "delete":
                return f"Remove-Item -Path '{file_path}' -Force"
            elif operation == "copy":
                dest = action_data.get("destination", "")
                return f"Copy-Item -Path '{file_path}' -Destination '{dest}' -Force"
        elif script_type == "python":
            if operation == "create":
                return f"Path('{file_path}').touch()"
            elif operation == "delete":
                return f"Path('{file_path}').unlink(missing_ok=True)"
            elif operation == "copy":
                dest = action_data.get("destination", "")
                return f"shutil.copy2('{file_path}', '{dest}')"
        
        return f"# File operation: {operation} on {file_path}"
    
    def _generate_command_execution(self, action_data: Dict[str, Any], script_type: str) -> str:
        """Generate command execution."""
        command = action_data.get("command", "")
        args = action_data.get("args", [])
        
        if isinstance(args, list):
            full_command = f"{command} {' '.join(args)}"
        else:
            full_command = f"{command} {args}"
        
        if script_type == "python":
            return f"subprocess.run(['{command}'] + {args}, check=True)"
        else:
            return full_command
    
    def _generate_build_operation_command(self, action_data: Dict[str, Any], script_type: str) -> str:
        """Generate build operation command."""
        build_tool = action_data.get("build_tool", "")
        target = action_data.get("target", "")
        
        if build_tool == "npm":
            return f"npm run {target}"
        elif build_tool == "maven":
            return f"mvn {target}"
        elif build_tool == "gradle":
            return f"./gradlew {target}"
        elif build_tool == "make":
            return f"make {target}"
        else:
            return f"{build_tool} {target}"
    
    def _generate_test_execution_command(self, action_data: Dict[str, Any], script_type: str) -> str:
        """Generate test execution command."""
        test_framework = action_data.get("test_framework", "")
        test_path = action_data.get("test_path", "")
        
        if test_framework == "pytest":
            return f"pytest {test_path}"
        elif test_framework == "jest":
            return f"npm test {test_path}"
        elif test_framework == "junit":
            return f"mvn test -Dtest={test_path}"
        else:
            return f"{test_framework} {test_path}"
    
    def _generate_script_configuration(
        self,
        pattern: ActionPattern,
        opportunity: AutomationOpportunity
    ) -> Dict[str, Any]:
        """Generate script configuration."""
        return {
            "pattern_id": str(pattern.id),
            "pattern_type": pattern.pattern_type.value,
            "opportunity_id": str(opportunity.id),
            "automation_score": opportunity.automation_score,
            "complexity": opportunity.complexity.value,
            "estimated_time_saving": opportunity.time_saving_potential,
            "frequency_per_week": opportunity.frequency_per_week,
            "risk_level": opportunity.risk_level,
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _get_environment_requirements(self, pattern: ActionPattern) -> List[str]:
        """Get environment requirements for the script."""
        requirements = []
        
        # Analyze actions to determine requirements
        actions = pattern.action_sequence
        for action in actions:
            action_type = action.get("type")
            action_data = action.get("data", {})
            
            if action_type == ActionType.BUILD_OPERATION.value:
                build_tool = action_data.get("build_tool", "")
                if build_tool and build_tool not in requirements:
                    requirements.append(build_tool)
            
            elif action_type == ActionType.COMMAND_EXECUTION.value:
                command = action_data.get("command", "")
                if command and command not in requirements:
                    requirements.append(command)
        
        return requirements
    
    async def _execute_script_safely(
        self,
        script: AutomationScript,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a script safely in a controlled environment."""
        try:
            # Create temporary file for script
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=f'.{script.script_type}',
                delete=False
            ) as temp_file:
                temp_file.write(script.script_content)
                temp_file_path = temp_file.name
            
            # Determine execution command
            if script.script_type == "bash":
                cmd = ["bash", temp_file_path]
            elif script.script_type == "python":
                cmd = ["python", temp_file_path]
            elif script.script_type == "powershell":
                cmd = ["powershell", "-File", temp_file_path]
            elif script.script_type == "batch":
                cmd = [temp_file_path]
            else:
                return {"success": False, "error": f"Unsupported script type: {script.script_type}"}
            
            # Execute with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=context.get("working_directory", ".")
            )
            
            # Clean up temporary file
            Path(temp_file_path).unlink(missing_ok=True)
            
            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "output": result.stdout,
                "error": result.stderr,
                "files_processed": self._count_files_processed(result.stdout)
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Script execution timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _count_files_processed(self, output: str) -> int:
        """Count files processed from script output (heuristic)."""
        # Simple heuristic - count lines that look like file operations
        lines = output.split('\n')
        file_lines = [line for line in lines if any(ext in line for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.h'])]
        return len(file_lines)
    
    def _get_bash_template(self) -> str:
        """Get bash script template."""
        return '''#!/bin/bash
# Generated automation script: {script_name}
# Description: {description}
# Generated at: {timestamp}

set -e  # Exit on error

echo "Starting automation: {script_name}"

{commands}

echo "Automation completed successfully"
'''
    
    def _get_python_template(self) -> str:
        """Get Python script template."""
        return '''#!/usr/bin/env python3
"""
Generated automation script: {script_name}
Description: {description}
Generated at: {timestamp}
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    """Main automation function."""
    print("Starting automation: {script_name}")
    
    try:
{commands}
        print("Automation completed successfully")
    except Exception as e:
        print(f"Automation failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _get_powershell_template(self) -> str:
        """Get PowerShell script template."""
        return '''# Generated automation script: {script_name}
# Description: {description}
# Generated at: {timestamp}

$ErrorActionPreference = "Stop"

Write-Host "Starting automation: {script_name}"

try {{
{commands}
    Write-Host "Automation completed successfully"
}} catch {{
    Write-Error "Automation failed: $($_.Exception.Message)"
    exit 1
}}
'''
    
    def _get_batch_template(self) -> str:
        """Get batch script template."""
        return '''@echo off
REM Generated automation script: {script_name}
REM Description: {description}
REM Generated at: {timestamp}

echo Starting automation: {script_name}

{commands}

echo Automation completed successfully
'''


# Global instance
workflow_generator = WorkflowGenerator()