# A framework for doing identity resolution at scale

Identity resolution enables organizations to link entities across a diversity of data sources, where entities can be
- Customers
- Products
- Employees
- Suppliers
- Etc.

This becomes challenging when an organization cannot, for one reason or another, use a consistent set of unique keys to identify their entities of interest. In this situation, we can resort to approximate matching, using a mix of human- and machine-powered to form the linkages. In doing so, the organization can answer questions such as:
- "Have I seen this entity before?"
- "What activity is associated with this entity?"

The framework here uses a set of open-source packages to enable search, approximate matching and linkage. It is intended to be run on Databricks Unified Analytics Platform using Delta Lake as an open-source storage layer, but could be adapted to run on any platform running Apache Spark. 