
from base import Base

class CCode(Base):

    """
    CCode(*lines, provides=..)
    """

    container_options = dict(CCodeLines=dict())

    template = '%(CCodeLines)s'

    def initialize(self, *lines, **options):
        self.lines = []
        map(self.add, lines)

    def update_containers(self):
        CCodeLines = self.get_container('CCodeLines')
        CCodeLines.add('\n'.join(self.lines))

    def add(self, component, label=None):
        if isinstance(component, str):
            assert label is None,`label`
            self.lines.append(component)
        elif isinstance(component, CCode):
            assert label is None,`label`
            self.lines.extend(component.lines)
        else:
            Base.add(self, component. label)

        
