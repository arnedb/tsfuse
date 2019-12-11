from graphviz import Graph

from tsfuse.errors import InvalidTagError

__all__ = [
    'Tags',
    'TagKey',
    'HierarchicalTagKey',
]


class Tags(object):
    """
    Mapping of tag keys to values.

    A tag is assigned to a unit of data and describes the contents of this data
    item. Each tag has a key and corresponding value. Possible tags are the
    sample rate, the sensor type and the location of the sensor. tsfuse
    defines a couple of standardized tags (see :class:`tsfuse.data.tags.TagKey`),
    but it is also possible to define your own custom tags.

    Parameters
    ----------
    tags : dict, optional
        Dictionary where each key is a :class:`tsfuse.data.tags.TagKey` instance
        and each value is in the domain of the key.

    Raises
    ------
    InvalidTagError
        If the given tag value is not valid for one of the keys.

    Examples
    --------
    >>> from tsfuse.data import Tags, TagKey
    >>> tags = Tags({TagKey.QUANTITY: 'acceleration'})
    """

    def __init__(self, tags=None):
        self._tags = {}
        self._properties = []
        if isinstance(tags, dict):
            for key in list(tags):
                self.add(key, tags[key])
        elif isinstance(tags, Tags):
            for key in tags.keys:
                self.add(key, tags[key])

    @property
    def keys(self):
        return list(self._tags)

    def add(self, key, value):
        """
        Add a new tag.

        Parameters
        ----------
        key : TagKey
            The key to identify the tag.
        value : int, float or str
            The value of the tag. The value should be valid, i.e., the value
            should be in the tag's domain.

        Raises
        ------
        InvalidTagException
            If the given value is not valid for the given key.

        Examples
        --------
        >>> tags = Tags()
        >>> tags.add(TagKey.QUANTITY, 'acceleration')
        """
        if isinstance(key, TagKey):
            if not key.is_valid(value):
                raise InvalidTagError()
        self._tags[key] = value
        if not hasattr(self, key.name):
            self._properties.append(key.name)
            setattr(self, key.name, value)

    def get(self, key):
        """
        Return the value for a given tag key.

        Parameters
        ----------
        key : TagKey or str
            The key of the tag to retrieve.

        Returns
        -------
        value
            Tag value or `None` if the tag key does not exist.
        """
        name = key.name if isinstance(key, TagKey) else key
        key = TagKey(name)
        if key in self._tags:
            return self._tags[key]
        else:
            return None

    def remove(self, key):
        """
        Remove a tag with a given key.

        Parameters
        ----------
        key : TagKey or str
            The key of the tag to remove.

        Raises
        ------
        KeyError
            If no tag with key `key` exists.
        """
        del self._tags[key]
        if key.name in self._properties:
            delattr(self, key.name)

    def check(self, *tags):
        """
        Check whether a set of tags is included in this collection.

        Parameters
        ----------
        *tags : (key, value)
            Key-value pairs to check for.

        Returns
        -------
        checked : boolean
            `True` if all key-value pairs exist.
        """
        for (key, value) in tags:
            if not self._tags.get(key) == value:
                return False
        return True

    def intersect(self, other):
        """
        Return the intersection of two Tags instances as a new Tags instance.
        The intersection consists of all (key, value) pairs that occur in
        both Tags instances.

        Parameters
        ----------
        other : Tags
            Other Tags instance.

        Returns
        -------
        intersection : Tags
        """
        intersection = Tags()
        for (key, value1) in iter(self):
            if key in other.keys:
                value2 = other[key]
                value = key.propagate(value1, value2)
                if value is not None:
                    intersection.add(key, value)
        return intersection

    def __iter__(self):
        return ((key, self._tags[key]) for key in self._tags)

    def __len__(self):
        return len(self._tags)

    def __contains__(self, key):
        name = key.name if isinstance(key, TagKey) else key
        return name in (tag.name for tag in self._tags)

    def __getitem__(self, key):
        return self.get(key)


class TagKey(object):
    """
    Name and domain of a tag.

    Parameters
    ----------
    name : str
        A name to identify the tag.
    domain : <class 'int'>, <class 'float'> or list(int or str), default <class 'int'>
        ``int``, ``float``, or a list of values.

    Attributes
    ----------
    QUANTITY : TagKey
        Default tag key specifying the quantity that a collection represents,
        see :ref:`quantity`.
    BODY_PART : HierarchicalTagKey
        Default tag key specifying the body part to which a sensor is attached,
        see :ref:`body-part`.
    """

    def __init__(self, name, domain=int):
        self.name = name
        self.domain = domain

    def propagate(self, value1, value2):
        if value1 == value2:
            return value1
        else:
            return None

    def is_valid(self, value):
        """
        Check whether a given value is allowed for the key.

        Parameters
        ----------
        value : object
            The value to check.

        Returns
        -------
        is_allowed : boolean
        """
        if self.domain == int:
            return isinstance(value, int)
        elif self.domain == float:
            return isinstance(value, int) or isinstance(value, float)
        elif isinstance(self.domain, list):
            return value in self.domain

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not (isinstance(other, TagKey)):
            return False
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class HierarchicalTagKey(TagKey):
    """
    Tag key where the domain has a lattice structure.

    Parameters
    ----------
    name : str
        Name of the tag key.
    lattice : tuple or list
        Domain specified as a lattice structure. The lattice must be specified as a tuple (or list)
        where each element is a tuple (or list) of length two: ``[specific, general]`` where
        ``specific`` is a value that is more specific than ``general`` (both values can be an
        ``int`` or ``str``).

    Examples
    --------
    >>> from tsfuse.data import HierarchicalTagKey
    >>> name = "location"
    >>> lattice = (('hand', 'body'), ('left', 'body'), ('hand_left', 'hand'), ('hand_left', 'left'))
    >>> location_tag_key = HierarchicalTagKey(name, lattice)
    """

    def __init__(self, name, lattice):
        values = []
        nodes = {}
        for specific, general in lattice:
            if specific not in values:
                values.append(specific)
                nodes[specific] = HierarchicalValue(specific)
            if general not in values:
                values.append(general)
                nodes[general] = HierarchicalValue(general)
            nodes[specific].add_parent(nodes[general])

        super(HierarchicalTagKey, self).__init__(name, lattice)
        self._values = values
        self._nodes = nodes
        self._propagated = dict()

    @property
    def lattice(self):
        return self.domain

    @property
    def values(self):
        return self._values

    def propagate(self, value1, value2):
        if (tuple(value1), tuple(value2)) in self._propagated:
            return self._propagated[(tuple(value1), tuple(value2))]

        if value1 in self.values:
            value1 = [value1]
        if value2 in self.values:
            value2 = [value2]

        value1 = sorted(value1)
        value2 = sorted(value2)

        # If equal: just return the same value
        if value1 == value2:
            if len(value1) == 1:
                return value1[0]
            else:
                return value1

        def intersect(sets):
            intersection = sets[0]
            for s in sets[1:]:
                intersection = intersection.intersection(s)
            return intersection

        # Otherwise: return least general generalization
        # TODO: optimize later
        values_more_general = []
        for v in value1 + value2:
            l = [v]
            for node in self._nodes[v].ancestors:
                l.append(node.value)
            values_more_general.append(set(l))
        values_more_general = intersect(values_more_general)
        lgg = []
        for v in values_more_general:
            if not any(v == a.value for g in values_more_general for a in self._nodes[g].ancestors):
                lgg.append(v)

        if len(lgg) == 0:
            self._propagated[(tuple(value1), tuple(value2))] = None
        elif len(lgg) == 1:
            self._propagated[(tuple(value1), tuple(value2))] = lgg[0]
        else:
            self._propagated[(tuple(value1), tuple(value2))] = sorted(lgg)

        return self._propagated[(tuple(value1), tuple(value2))]

    def is_valid(self, value):
        if value in self.values:
            return True
        elif isinstance(value, (list, tuple)):
            if all(v in self.values for v in value):
                return True
        return False

    def _repr_svg_(self):
        dot = Graph()
        values = []
        for specific, general in self.lattice:
            if specific not in values:
                values.append(specific)
                dot.node(specific)
            if general not in values:
                values.append(general)
                dot.node(general)
            dot.edge(general, specific)
        return dot._repr_svg_()


def common_tags(data):
    """
    Return a `Tags` instance containing all common tags of a given list of
    data collections, i.e., the tags for which all data collections have the
    same value.

    Parameters
    ----------
    data : list(DataCollection)
        List of DataCollection instances.

    Returns
    -------
    Tags
    """
    if len(data) == 0:
        return Tags()
    else:
        tags = data[0].tags
        for d in data[1:]:
            tags = tags.intersect(d.tags)
        return tags


class HierarchicalValue(object):
    def __init__(self, value):
        self.value = value
        self.parents = []
        self._ancestors = None

    def add_parent(self, parent):
        if parent not in self.parents:
            self.parents.append(parent)

    @property
    def ancestors(self):
        if self._ancestors is None:
            l = self.parents
            for p in self.parents:
                l += p.ancestors
            self._ancestors = l
            return l
        else:
            return self._ancestors

    def __eq__(self, other):
        if not isinstance(other, HierarchicalTagKey):
            return False
        else:
            return self.value == other.values

    def __hash__(self):
        return hash(self.value)


####################################################################################################
# Pre-defined tag keys
####################################################################################################


TagKey.QUANTITY = TagKey('quantity', [
    'jerk',
    'acceleration',
    'velocity',
    'position',
    'angular_velocity',
    'angle',
    'magnetic_field',
])
quantity = TagKey.QUANTITY

body_part_lattice = [
    ('head', 'upper'),
    ('neck', 'upper'),
    ('back', 'upper'),
    ('arm', 'upper'),
    ('leg', 'lower'),
]

for side in ['left', 'right']:
    body_part_lattice += [
        ('arm_{}'.format(side), '{}'.format(side)),
        ('arm_{}'.format(side), 'arm'),
        ('leg_{}'.format(side), '{}'.format(side)),
        ('leg_{}'.format(side), 'leg'),
        ('shoulder_{}'.format(side), 'arm_left'.format(side)),
        ('shoulder_{}'.format(side), 'shoulder'),
        ('elbow_{}'.format(side), 'arm_{}'.format(side)),
        ('elbow_{}'.format(side), 'elbow'),
        ('hand_{}'.format(side), 'arm_{}'.format(side)),
        ('hand_{}'.format(side), 'hand'),
        ('wrist_{}'.format(side), 'arm_{}'.format(side)),
        ('wrist_{}'.format(side), 'wrist'),
        ('hip_{}'.format(side), 'leg_{}'.format(side)),
        ('hip_{}'.format(side), 'hip'),
        ('knee_{}'.format(side), 'leg_{}'.format(side)),
        ('knee_{}'.format(side), 'knee'),
        ('foot_{}'.format(side), 'leg_{}'.format(side)),
        ('foot_{}'.format(side), 'foot'),
        ('ankle_{}'.format(side), 'leg_{}'.format(side)),
        ('ankle_{}'.format(side), 'ankle'),
    ]

TagKey.BODY_PART = HierarchicalTagKey('body_part', body_part_lattice)
body_part = TagKey.BODY_PART
